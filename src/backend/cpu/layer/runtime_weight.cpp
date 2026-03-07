#include "runtime_weight.h"

#include "minfer/basic_op.h"
#include "minfer/system.h"

namespace minfer
{

void RuntimeWeight::init(const Mat& fp32_weight, Int8QuantScheme scheme)
{
    scheme_ = scheme;
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
}

void RuntimeWeight::setPrecision(RuntimePrecision precision)
{
    precision_ = precision;
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
