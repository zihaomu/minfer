#ifndef MINFER_NORMALIZATION_KERNEL_HWY_H
#define MINFER_NORMALIZATION_KERNEL_HWY_H

#include <cstddef>

namespace minfer {
namespace cpu {

void softmax_lastdim_hwy(const float* input, float* output, size_t outer, size_t inner);

void rmsnorm_lastdim_hwy(const float* input,
                         const float* weight,
                         float* output,
                         size_t outer,
                         size_t channels,
                         float eps);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_NORMALIZATION_KERNEL_HWY_H
