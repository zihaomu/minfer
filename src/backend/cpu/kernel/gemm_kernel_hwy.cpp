#include "gemm_kernel_hwy.h"

#include "hwy/highway.h"

namespace minfer {
namespace cpu {

namespace hn = hwy::HWY_NAMESPACE;

void gemm_kernel_hwy_nn(const float* a, const float* b, float* c,
                        int m, int n, int k) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));

    for (int mi = 0; mi < m; ++mi) {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        int ni = 0;
        for (; ni + lanes <= n; ni += lanes) {
            auto sum = hn::Zero(d);
            for (int ki = 0; ki < k; ++ki) {
                const auto vb = hn::LoadU(d, b + static_cast<size_t>(ki) * n + ni);
                sum = hn::MulAdd(hn::Set(d, a_row[ki]), vb, sum);
            }
            hn::StoreU(sum, d, c_row + ni);
        }

        for (; ni < n; ++ni) {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki) {
                sum += a_row[ki] * b[static_cast<size_t>(ki) * n + ni];
            }
            c_row[ni] = sum;
        }
    }
}

void gemm_kernel_hwy_nt(const float* a, const float* b, float* c,
                        int m, int n, int k) {
    const hn::ScalableTag<float> d;
    const int lanes = static_cast<int>(hn::Lanes(d));

    for (int mi = 0; mi < m; ++mi) {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni) {
            const float* b_row = b + static_cast<size_t>(ni) * k;

            auto sum_vec = hn::Zero(d);
            int ki = 0;
            for (; ki + lanes <= k; ki += lanes) {
                const auto va = hn::LoadU(d, a_row + ki);
                const auto vb = hn::LoadU(d, b_row + ki);
                sum_vec = hn::MulAdd(va, vb, sum_vec);
            }

            float sum = hn::ReduceSum(d, sum_vec);
            for (; ki < k; ++ki) {
                sum += a_row[ki] * b_row[ki];
            }

            c_row[ni] = sum;
        }
    }
}

}  // namespace cpu
}  // namespace minfer
