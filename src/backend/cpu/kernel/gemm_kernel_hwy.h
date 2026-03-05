#ifndef MINFER_GEMM_KERNEL_HWY_H
#define MINFER_GEMM_KERNEL_HWY_H

namespace minfer {
namespace cpu {

// A[M, K] x B[K, N] -> C[M, N]
void gemm_kernel_hwy_nn(const float* a, const float* b, float* c,
                        int m, int n, int k);

// A[M, K] x B[N, K] -> C[M, N], where B is row-major [N, K].
void gemm_kernel_hwy_nt(const float* a, const float* b, float* c,
                        int m, int n, int k);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_GEMM_KERNEL_HWY_H
