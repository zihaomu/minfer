#ifndef MINFER_GEMM_KERNEL_XSIMD_H
#define MINFER_GEMM_KERNEL_XSIMD_H

#include "minfer/define.h"

namespace minfer {
namespace cpu {

// A[M, K] x B[K, N] -> C[M, N]
void gemm_kernel_xsimd_nn(const float* a, const float* b, float* c,
                          int m, int n, int k);

// A[M, K] x B[N, K] -> C[M, N], where B is row-major [N, K].
void gemm_kernel_xsimd_nt(const float* a, const float* b, float* c,
                          int m, int n, int k);

void gemm_kernel_xsimd_nn_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k);

void gemm_kernel_xsimd_nt_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k);

void gemm_kernel_xsimd_nt_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_GEMM_KERNEL_XSIMD_H
