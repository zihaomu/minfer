#include "minfer/mat.h"
#include "minfer/system.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace minfer
{

namespace
{

void ensure_fp32_mat(const Mat& src)
{
    M_Assert(!src.empty() && "Quantization source mat must not be empty!");
    M_Assert(src.type() == DT_32F && "Quantization currently expects FP32 source data!");
}

}  // namespace

void quantize_int8_per_tensor(const Mat& src, Mat& dst_int8, Mat& dst_scales)
{
    ensure_fp32_mat(src);

    dst_int8.create(src.dims, src.size.p, DT_8S);
    dst_scales.create({1}, DT_32F);

    const float* src_data = reinterpret_cast<const float*>(src.data);
    int8_t* dst_data = reinterpret_cast<int8_t*>(dst_int8.data);
    float* scale_ptr = reinterpret_cast<float*>(dst_scales.data);

    float max_abs = 0.0f;
    for (size_t i = 0; i < src.total(); ++i)
    {
        max_abs = std::max(max_abs, std::abs(src_data[i]));
    }

    const float scale = max_abs > std::numeric_limits<float>::epsilon() ? max_abs / 127.0f : 1.0f;
    scale_ptr[0] = scale;

    for (size_t i = 0; i < src.total(); ++i)
    {
        const float q = std::round(src_data[i] / scale);
        dst_data[i] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, q)));
    }
}

void quantize_int8_per_row(const Mat& src, Mat& dst_int8, Mat& dst_scales)
{
    ensure_fp32_mat(src);
    M_Assert(src.dims >= 1);

    dst_int8.create(src.dims, src.size.p, DT_8S);
    const int rows = src.size[0];
    const size_t row_size = src.total() / rows;
    dst_scales.create({rows}, DT_32F);

    const float* src_data = reinterpret_cast<const float*>(src.data);
    int8_t* dst_data = reinterpret_cast<int8_t*>(dst_int8.data);
    float* scales = reinterpret_cast<float*>(dst_scales.data);

    for (int row = 0; row < rows; ++row)
    {
        const float* src_row = src_data + static_cast<size_t>(row) * row_size;
        int8_t* dst_row = dst_data + static_cast<size_t>(row) * row_size;

        float max_abs = 0.0f;
        for (size_t col = 0; col < row_size; ++col)
        {
            max_abs = std::max(max_abs, std::abs(src_row[col]));
        }

        const float scale = max_abs > std::numeric_limits<float>::epsilon() ? max_abs / 127.0f : 1.0f;
        scales[row] = scale;

        for (size_t col = 0; col < row_size; ++col)
        {
            const float q = std::round(src_row[col] / scale);
            dst_row[col] = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, q)));
        }
    }
}

void dequantize_int8_per_tensor(const Mat& src_int8, const Mat& scales, Mat& dst_fp32)
{
    M_Assert(src_int8.type() == DT_8S);
    M_Assert(scales.type() == DT_32F && scales.total() == 1);

    dst_fp32.create(src_int8.dims, src_int8.size.p, DT_32F);
    const int8_t* src_data = reinterpret_cast<const int8_t*>(src_int8.data);
    const float scale = reinterpret_cast<const float*>(scales.data)[0];
    float* dst_data = reinterpret_cast<float*>(dst_fp32.data);

    for (size_t i = 0; i < src_int8.total(); ++i)
    {
        dst_data[i] = static_cast<float>(src_data[i]) * scale;
    }
}

void dequantize_int8_per_row(const Mat& src_int8, const Mat& scales, Mat& dst_fp32)
{
    M_Assert(src_int8.type() == DT_8S);
    M_Assert(scales.type() == DT_32F);
    M_Assert(src_int8.dims >= 1);
    M_Assert(scales.total() == src_int8.size[0]);

    dst_fp32.create(src_int8.dims, src_int8.size.p, DT_32F);
    const int rows = src_int8.size[0];
    const size_t row_size = src_int8.total() / rows;
    const int8_t* src_data = reinterpret_cast<const int8_t*>(src_int8.data);
    const float* scale_data = reinterpret_cast<const float*>(scales.data);
    float* dst_data = reinterpret_cast<float*>(dst_fp32.data);

    for (int row = 0; row < rows; ++row)
    {
        const int8_t* src_row = src_data + static_cast<size_t>(row) * row_size;
        float* dst_row = dst_data + static_cast<size_t>(row) * row_size;
        const float scale = scale_data[row];
        for (size_t col = 0; col < row_size; ++col)
        {
            dst_row[col] = static_cast<float>(src_row[col]) * scale;
        }
    }
}

}  // namespace minfer
