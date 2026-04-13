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

    void init(const Mat& fp32_weight,
              Int8QuantScheme scheme = Int8QuantScheme::PerTensor,
              bool enable_decode_pack = false,
              RuntimePrecision precision = RuntimePrecision::FP32);
    void setPrecision(RuntimePrecision precision);

    RuntimePrecision precision() const;
    bool empty() const;
    bool usesInt8() const;
    bool hasDecodePacked() const;
    bool shouldUseDecodePacked(const Mat& input) const;

    const Mat& active() const;
    const Mat& int8Scales() const;
    const Mat& fp32() const;
    const Mat& decodePacked() const;
    Mat materializeFp32() const;
    void gemmNT(const Mat& input, Mat& output) const;
    Mat gemmNT(const Mat& input) const;
    bool gemmNTPair(const Mat& input, const RuntimeWeight& other, Mat& out0, Mat& out1) const;
    bool selectNT(const Mat& input,
                  DecodeOutputMode mode,
                  int top_k,
                  const Mat* bias,
                  DecodeSelection& selection) const;

private:
    void rebuildDecodePacked();

    Int8QuantScheme scheme_ = Int8QuantScheme::PerTensor;
    RuntimePrecision precision_ = RuntimePrecision::FP32;
    bool enable_decode_pack_ = false;
    Mat weight_;
    Mat int8_scales_;
    Mat decode_kernel_packed_;
    Mat decode_packed_scales_;
};

Mat canonicalize_linear_weight(const Mat& weight, int out_features, int in_features);
Mat canonicalize_lookup_weight(const Mat& weight, int rows, int cols);

}  // namespace minfer

#endif  // MINFER_RUNTIME_WEIGHT_H
