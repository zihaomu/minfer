#ifndef MINFER_RUNTIME_WEIGHT_H
#define MINFER_RUNTIME_WEIGHT_H

#include "minfer/context.h"
#include "minfer/mat.h"

namespace minfer
{

enum class Int8QuantScheme {
    PerTensor,
    PerRow,
};

class RuntimeWeight
{
public:
    RuntimeWeight() = default;

    void init(const Mat& fp32_weight, Int8QuantScheme scheme = Int8QuantScheme::PerTensor);
    void setPrecision(RuntimePrecision precision);

    RuntimePrecision precision() const;
    bool empty() const;
    bool usesInt8() const;

    const Mat& active() const;
    const Mat& int8Scales() const;
    const Mat& fp32() const;
    Mat materializeFp32() const;

private:
    Int8QuantScheme scheme_ = Int8QuantScheme::PerTensor;
    RuntimePrecision precision_ = RuntimePrecision::FP32;
    Mat fp32_;
    Mat fp16_;
    Mat int8_;
    Mat int8_scales_;
};

Mat canonicalize_linear_weight(const Mat& weight, int out_features, int in_features);
Mat canonicalize_lookup_weight(const Mat& weight, int rows, int cols);

}  // namespace minfer

#endif  // MINFER_RUNTIME_WEIGHT_H
