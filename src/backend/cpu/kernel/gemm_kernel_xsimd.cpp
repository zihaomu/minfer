#include "gemm_kernel_xsimd.h"
#include "openmp_utils.h"
#include "xsimd_kernel_utils.h"

#include "xsimd/xsimd.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace {

inline float dot_fp32_xsimd(const float* a_row, const float* b_row, int k)
{
    XSimdBatch sum_vec(0.0f);
    int ki = 0;
    for (; ki + static_cast<int>(kXSimdBatchSize) <= k; ki += static_cast<int>(kXSimdBatchSize))
    {
        const XSimdBatch va = XSimdBatch::load_unaligned(a_row + ki);
        const XSimdBatch vb = XSimdBatch::load_unaligned(b_row + ki);
        sum_vec = xsimd::fma(va, vb, sum_vec);
    }

    float sum = xsimd::reduce_add(sum_vec);
    for (; ki < k; ++ki)
    {
        sum += a_row[ki] * b_row[ki];
    }
    return sum;
}

inline float dot_fp16_xsimd(const float* a_row, const hfloat* b_row, int k)
{
    XSimdBatch sum_vec(0.0f);
    int ki = 0;
    for (; ki + static_cast<int>(kXSimdBatchSize) <= k; ki += static_cast<int>(kXSimdBatchSize))
    {
        const XSimdBatch va = XSimdBatch::load_unaligned(a_row + ki);
        const XSimdBatch vb = load_hfloat_batch(b_row + ki);
        sum_vec = xsimd::fma(va, vb, sum_vec);
    }

    float sum = xsimd::reduce_add(sum_vec);
    for (; ki < k; ++ki)
    {
        sum += a_row[ki] * static_cast<float>(b_row[ki]);
    }
    return sum;
}

inline float dot_i8_rowwise_xsimd(const float* a_row, const int8_t* b_row, float scale, int k)
{
    XSimdBatch sum_vec(0.0f);
    const XSimdBatch scale_vec(scale);
    int ki = 0;
    for (; ki + static_cast<int>(kXSimdBatchSize) <= k; ki += static_cast<int>(kXSimdBatchSize))
    {
        const XSimdBatch va = XSimdBatch::load_unaligned(a_row + ki);
        const XSimdBatch vb = load_int8_batch(b_row + ki) * scale_vec;
        sum_vec = xsimd::fma(va, vb, sum_vec);
    }

    float sum = xsimd::reduce_add(sum_vec);
    for (; ki < k; ++ki)
    {
        sum += a_row[ki] * (static_cast<float>(b_row[ki]) * scale);
    }
    return sum;
}

}  // namespace

void gemm_kernel_xsimd_nn(const float* a, const float* b, float* c,
                          int m, int n, int k)
{
    const int lanes = static_cast<int>(kXSimdBatchSize);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        int ni = 0;
        for (; ni + lanes <= n; ni += lanes)
        {
            XSimdBatch sum_vec(0.0f);
            for (int ki = 0; ki < k; ++ki)
            {
                const XSimdBatch a_vec(a_row[ki]);
                const XSimdBatch b_vec = XSimdBatch::load_unaligned(b + static_cast<size_t>(ki) * n + ni);
                sum_vec = xsimd::fma(a_vec, b_vec, sum_vec);
            }
            sum_vec.store_unaligned(c_row + ni);
        }

        for (; ni < n; ++ni)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                sum += a_row[ki] * b[static_cast<size_t>(ki) * n + ni];
            }
            c_row[ni] = sum;
        }
    }
}

void gemm_kernel_xsimd_nt(const float* a, const float* b, float* c,
                          int m, int n, int k)
{
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni)
        {
            const float* b_row = b + static_cast<size_t>(ni) * k;
            c_row[ni] = dot_fp32_xsimd(a_row, b_row, k);
        }
    }
}

void gemm_kernel_xsimd_nn_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k)
{
    const int lanes = static_cast<int>(kXSimdBatchSize);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        int ni = 0;
        for (; ni + lanes <= n; ni += lanes)
        {
            XSimdBatch sum_vec(0.0f);
            for (int ki = 0; ki < k; ++ki)
            {
                const XSimdBatch a_vec(a_row[ki]);
                const XSimdBatch b_vec = load_hfloat_batch(b + static_cast<size_t>(ki) * n + ni);
                sum_vec = xsimd::fma(a_vec, b_vec, sum_vec);
            }
            sum_vec.store_unaligned(c_row + ni);
        }

        for (; ni < n; ++ni)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                sum += a_row[ki] * static_cast<float>(b[static_cast<size_t>(ki) * n + ni]);
            }
            c_row[ni] = sum;
        }
    }
}

void gemm_kernel_xsimd_nt_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k)
{
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni)
        {
            const hfloat* b_row = b + static_cast<size_t>(ni) * k;
            c_row[ni] = dot_fp16_xsimd(a_row, b_row, k);
        }
    }
}

void gemm_kernel_xsimd_nt_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k)
{
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(m, static_cast<size_t>(n) * static_cast<size_t>(k), 1LL << 16, 1))
#endif
    for (int mi = 0; mi < m; ++mi)
    {
        const float* a_row = a + static_cast<size_t>(mi) * k;
        float* c_row = c + static_cast<size_t>(mi) * n;

        for (int ni = 0; ni < n; ++ni)
        {
            const int8_t* b_row = b + static_cast<size_t>(ni) * k;
            c_row[ni] = dot_i8_rowwise_xsimd(a_row, b_row, scales[ni], k);
        }
    }
}

}  // namespace cpu
}  // namespace minfer
