#ifndef MINFER_ACTIVATION_KERNEL_XSIMD_H
#define MINFER_ACTIVATION_KERNEL_XSIMD_H

#include <cstddef>

namespace minfer {
namespace cpu {

void silu_kernel_xsimd(const float* input, float* output, size_t count);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_ACTIVATION_KERNEL_XSIMD_H
