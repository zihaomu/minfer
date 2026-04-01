#ifndef MINFER_BINARY_KERNEL_XSIMD_H
#define MINFER_BINARY_KERNEL_XSIMD_H

#include <cstddef>
#include <cstdint>

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

void binary_add_weighted_xsimd(const float* lhs,
                               const float* rhs,
                               float* out,
                               size_t total,
                               float alpha,
                               float beta);
void binary_add_weighted_xsimd(const int32_t* lhs,
                               const int32_t* rhs,
                               int32_t* out,
                               size_t total,
                               float alpha,
                               float beta);

void unary_negate_xsimd(const float* src, float* dst, size_t total);
void unary_negate_xsimd(const int32_t* src, int32_t* dst, size_t total);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_BINARY_KERNEL_XSIMD_H
