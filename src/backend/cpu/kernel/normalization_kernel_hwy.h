#ifndef MINFER_NORMALIZATION_KERNEL_HWY_H
#define MINFER_NORMALIZATION_KERNEL_HWY_H

#include "minfer/define.h"

#include <cstddef>
#include <cstdint>

namespace minfer {
namespace cpu {

void softmax_lastdim_hwy(const float* input, float* output, size_t outer, size_t inner);

void causal_masked_softmax_square_hwy(const float* input,
                                      float* output,
                                      size_t outer,
                                      size_t seq_len,
                                      float scale = 1.0f);

void rmsnorm_lastdim_hwy_fp16_weight(const float* input,
                                     const hfloat* weight,
                                     float* output,
                                     size_t outer,
                                     size_t channels,
                                     float eps);

void rmsnorm_lastdim_hwy_i8_weight(const float* input,
                                   const int8_t* weight,
                                   const float* scales,
                                   float* output,
                                   size_t outer,
                                   size_t channels,
                                   float eps);

void rmsnorm_lastdim_hwy(const float* input,
                         const float* weight,
                         float* output,
                         size_t outer,
                         size_t channels,
                         float eps);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_NORMALIZATION_KERNEL_HWY_H
