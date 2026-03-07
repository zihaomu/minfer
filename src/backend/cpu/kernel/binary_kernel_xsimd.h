#ifndef MINFER_BINARY_KERNEL_XSIMD_H
#define MINFER_BINARY_KERNEL_XSIMD_H

#include <cstddef>

namespace minfer {
namespace cpu {

enum class BinaryKernelOp
{
    Add = 0,
    Sub,
    Mul,
    Div,
};

void binary_broadcast_xsimd(BinaryKernelOp op,
                            const float* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const float* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            float* out,
                            size_t outer,
                            size_t inner);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_BINARY_KERNEL_XSIMD_H
