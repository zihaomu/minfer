#include "runtime_weight.h"

#include "backend/cpu/kernel/gemm_kernel_xsimd.h"
#include "backend/cpu/kernel/openmp_utils.h"
#include "minfer/basic_op.h"
#include "minfer/system.h"

#include <vector>

namespace minfer
{

void RuntimeWeight::init(const Mat& fp32_weight, Int8QuantScheme scheme, bool enable_decode_pack)
{
    scheme_ = scheme;
    enable_decode_pack_ = enable_decode_pack;
    fp32_weight.convertTo(fp32_, DT_32F);
    fp32_.convertTo(fp16_, DT_16F);
    if (scheme_ == Int8QuantScheme::PerRow)
    {
        quantize_int8_per_row(fp32_, int8_, int8_scales_);
    }
    else
    {
        quantize_int8_per_tensor(fp32_, int8_, int8_scales_);
    }
    precision_ = RuntimePrecision::FP32;
    rebuildDecodePacked();
}

void RuntimeWeight::setPrecision(RuntimePrecision precision)
{
    precision_ = precision;
    rebuildDecodePacked();
}

RuntimePrecision RuntimeWeight::precision() const
{
    return precision_;
}

bool RuntimeWeight::empty() const
{
    return fp32_.empty();
}

bool RuntimeWeight::usesInt8() const
{
    return precision_ == RuntimePrecision::INT8;
}

bool RuntimeWeight::hasDecodePacked() const
{
    return !decode_packed_.empty();
}

const Mat& RuntimeWeight::active() const
{
    switch (precision_)
    {
        case RuntimePrecision::FP16:
            return fp16_;
        case RuntimePrecision::INT8:
            return int8_;
        case RuntimePrecision::FP32:
        default:
            return fp32_;
    }
}

const Mat& RuntimeWeight::int8Scales() const
{
    return int8_scales_;
}

const Mat& RuntimeWeight::fp32() const
{
    return fp32_;
}

const Mat& RuntimeWeight::decodePacked() const
{
    return decode_packed_;
}

Mat RuntimeWeight::materializeFp32() const
{
    switch (precision_)
    {
        case RuntimePrecision::FP16:
        {
            Mat out;
            fp16_.convertTo(out, DT_32F);
            return out;
        }
        case RuntimePrecision::INT8:
        {
            Mat out;
            if (scheme_ == Int8QuantScheme::PerRow)
            {
                dequantize_int8_per_row(int8_, int8_scales_, out);
            }
            else
            {
                dequantize_int8_per_tensor(int8_, int8_scales_, out);
            }
            return out;
        }
        case RuntimePrecision::FP32:
        default:
            return fp32_;
    }
}

void RuntimeWeight::rebuildDecodePacked()
{
    decode_packed_.release();
    decode_kernel_packed_.release();
    decode_packed_scales_.release();
    if (!enable_decode_pack_ || fp32_.empty())
    {
        return;
    }

    M_Assert(fp32_.shape().size() == 2 && "Decode-packed weights require a 2D matrix");

    const Mat* src = &fp32_;
    switch (precision_)
    {
        case RuntimePrecision::FP16:
            src = &fp16_;
            break;
        case RuntimePrecision::INT8:
            src = &int8_;
            break;
        case RuntimePrecision::FP32:
        default:
            src = &fp32_;
            break;
    }

    decode_packed_ = transposeND(*src, {1, 0});

    const int K = decode_packed_.shape()[0];
    const int N = decode_packed_.shape()[1];
    const std::vector<int> packed_b_shape = {static_cast<int>(cpu::gemm_xsimd_packed_b_elements(N, K))};
    const std::vector<int> packed_scale_shape = {static_cast<int>(cpu::gemm_xsimd_packed_scale_elements(N))};

    switch (precision_)
    {
        case RuntimePrecision::FP16:
            decode_kernel_packed_.create(packed_b_shape, DT_16F);
            cpu::gemm_pack_xsimd_nn_fp16(reinterpret_cast<const hfloat*>(decode_packed_.data),
                                         reinterpret_cast<hfloat*>(decode_kernel_packed_.data),
                                         N,
                                         K);
            break;
        case RuntimePrecision::INT8:
        {
            decode_kernel_packed_.create(packed_b_shape, DT_8S);
            decode_packed_scales_.create(packed_scale_shape, DT_32F);

            if (scheme_ == Int8QuantScheme::PerRow)
            {
                cpu::gemm_pack_xsimd_nn_i8_rowwise(reinterpret_cast<const int8_t*>(decode_packed_.data),
                                                   reinterpret_cast<const float*>(int8_scales_.data),
                                                   reinterpret_cast<int8_t*>(decode_kernel_packed_.data),
                                                   reinterpret_cast<float*>(decode_packed_scales_.data),
                                                   N,
                                                   K);
            }
            else
            {
                std::vector<float> repeated_scales(static_cast<size_t>(N),
                                                   reinterpret_cast<const float*>(int8_scales_.data)[0]);
                cpu::gemm_pack_xsimd_nn_i8_rowwise(reinterpret_cast<const int8_t*>(decode_packed_.data),
                                                   repeated_scales.data(),
                                                   reinterpret_cast<int8_t*>(decode_kernel_packed_.data),
                                                   reinterpret_cast<float*>(decode_packed_scales_.data),
                                                   N,
                                                   K);
            }
            break;
        }
        case RuntimePrecision::FP32:
        default:
            decode_kernel_packed_.create(packed_b_shape, DT_32F);
            cpu::gemm_pack_xsimd_nn_fp32(reinterpret_cast<const float*>(decode_packed_.data),
                                         reinterpret_cast<float*>(decode_kernel_packed_.data),
                                         N,
                                         K);
            break;
    }
}

Mat RuntimeWeight::gemmNT(const Mat& input) const
{
    if (!hasDecodePacked() || decode_kernel_packed_.empty())
    {
        if (usesInt8())
        {
            return gemm(input, active(), int8Scales(), false, true);
        }
        return gemm(input, active(), false, true);
    }

    const MatShape in_shape = input.shape();
    M_Assert(in_shape.size() >= 2);
    const int matrix_rows = in_shape[in_shape.size() - 2];
    if (matrix_rows != 1)
    {
        if (usesInt8())
        {
            return gemm(input, active(), int8Scales(), false, true);
        }
        return gemm(input, active(), false, true);
    }

    const int K = decode_packed_.shape()[0];
    const int N = decode_packed_.shape()[1];
    M_Assert(in_shape.back() == K);

    MatShape out_shape = in_shape;
    out_shape.back() = N;
    Mat out(out_shape, DT_32F);

    const size_t outer = input.total() / static_cast<size_t>(K);
    const float* input_ptr = reinterpret_cast<const float*>(input.data);
    float* output_ptr = reinterpret_cast<float*>(out.data);

    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(cpu::should_parallelize_1d_loop(outer, static_cast<size_t>(N) * static_cast<size_t>(K), 1LL << 16, 2))
#endif
    for (long long idx_ll = 0; idx_ll < outer_ll; ++idx_ll)
    {
        const size_t idx = static_cast<size_t>(idx_ll);
        const float* row_in = input_ptr + idx * static_cast<size_t>(K);
        float* row_out = output_ptr + idx * static_cast<size_t>(N);
        switch (precision_)
        {
            case RuntimePrecision::FP16:
                cpu::gemm_kernel_xsimd_row_packed_fp16(row_in,
                                                       reinterpret_cast<const hfloat*>(decode_kernel_packed_.data),
                                                       row_out,
                                                       N,
                                                       K);
                break;
            case RuntimePrecision::INT8:
                cpu::gemm_kernel_xsimd_row_packed_i8_rowwise(row_in,
                                                             reinterpret_cast<const int8_t*>(decode_kernel_packed_.data),
                                                             reinterpret_cast<const float*>(decode_packed_scales_.data),
                                                             row_out,
                                                             N,
                                                             K);
                break;
            case RuntimePrecision::FP32:
            default:
                cpu::gemm_kernel_xsimd_row_packed_fp32(row_in,
                                                       reinterpret_cast<const float*>(decode_kernel_packed_.data),
                                                       row_out,
                                                       N,
                                                       K);
                break;
        }
    }

    return out;
}

Mat canonicalize_linear_weight(const Mat& weight, int out_features, int in_features)
{
    Mat weight_fp32;
    weight.convertTo(weight_fp32, DT_32F);

    MatShape shape = weight_fp32.shape();
    M_Assert(shape.size() == 2);
    if (shape[0] == out_features && shape[1] == in_features)
    {
        return weight_fp32;
    }
    if (shape[0] == in_features && shape[1] == out_features)
    {
        return transposeND(weight_fp32, {1, 0});
    }
    M_Error_(Error::StsBadSize, ("Weight shape is not compatible with out=%d, in=%d", out_features, in_features));
    return {};
}

Mat canonicalize_lookup_weight(const Mat& weight, int rows, int cols)
{
    Mat weight_fp32;
    weight.convertTo(weight_fp32, DT_32F);

    MatShape shape = weight_fp32.shape();
    M_Assert(shape.size() == 2);
    if (shape[0] == rows && shape[1] == cols)
    {
        return weight_fp32;
    }
    if (shape[0] == cols && shape[1] == rows)
    {
        return transposeND(weight_fp32, {1, 0});
    }
    M_Error_(Error::StsBadSize, ("Lookup weight shape is not compatible with rows=%d, cols=%d", rows, cols));
    return {};
}

}  // namespace minfer
