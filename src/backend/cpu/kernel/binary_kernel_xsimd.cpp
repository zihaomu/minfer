#include "binary_kernel_xsimd.h"
#include "openmp_utils.h"

#include "xsimd/xsimd.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace {

using Batch = xsimd::batch<float>;
constexpr size_t kLanes = Batch::size;

inline float apply_scalar(BinaryKernelOp op, float lhs, float rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
    }

    return 0.0f;
}

inline Batch apply_batch(BinaryKernelOp op, const Batch& lhs, const Batch& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
    }

    return Batch(0.0f);
}

}  // namespace

void binary_broadcast_xsimd(BinaryKernelOp op,
                            const float* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const float* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            float* out,
                            size_t outer,
                            size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* lhs_row = lhs + outer_i * lhs_outer_stride;
        const float* rhs_row = rhs + outer_i * rhs_outer_stride;
        float* out_row = out + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0
                    ? Batch(lhs_row[0])
                    : Batch::load_unaligned(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0
                    ? Batch(rhs_row[0])
                    : Batch::load_unaligned(rhs_row + inner_idx);
                const Batch out_vec = apply_batch(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const float rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar(op, lhs_val, rhs_val);
        }
    }
}

void binary_add_weighted_xsimd(const float* lhs,
                               const float* rhs,
                               float* out,
                               size_t total,
                               float alpha,
                               float beta)
{
    const Batch alpha_batch(alpha);
    const Batch beta_batch(beta);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(total / kLanes, kLanes, 1LL << 16, 2))
#endif
    for (long long i = 0; i <= static_cast<long long>(total) - static_cast<long long>(kLanes); i += static_cast<long long>(kLanes))
    {
        const auto idx = static_cast<size_t>(i);
        const Batch a = Batch::load_unaligned(lhs + idx);
        const Batch b = Batch::load_unaligned(rhs + idx);
        const Batch y = xsimd::fma(a, alpha_batch, b * beta_batch);
        y.store_unaligned(out + idx);
    }

    const size_t vec_end = (total / kLanes) * kLanes;
    for (size_t i = vec_end; i < total; ++i)
    {
        out[i] = lhs[i] * alpha + rhs[i] * beta;
    }
}

void binary_add_weighted_xsimd(const int32_t* lhs,
                               const int32_t* rhs,
                               int32_t* out,
                               size_t total,
                               float alpha,
                               float beta)
{
    using IntBatch = xsimd::batch<int32_t>;
    using FloatBatch = xsimd::batch<float>;
    constexpr size_t int_lanes = IntBatch::size;
    const FloatBatch alpha_batch(alpha);
    const FloatBatch beta_batch(beta);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(total / int_lanes, int_lanes, 1LL << 16, 2))
#endif
    for (long long i = 0; i <= static_cast<long long>(total) - static_cast<long long>(int_lanes); i += static_cast<long long>(int_lanes))
    {
        const auto idx = static_cast<size_t>(i);
        const IntBatch a_i = IntBatch::load_unaligned(lhs + idx);
        const IntBatch b_i = IntBatch::load_unaligned(rhs + idx);
        const FloatBatch a = xsimd::to_float(a_i);
        const FloatBatch b = xsimd::to_float(b_i);
        const FloatBatch y = xsimd::fma(a, alpha_batch, b * beta_batch);
        const IntBatch y_i = xsimd::to_int(xsimd::trunc(y));
        y_i.store_unaligned(out + idx);
    }

    const size_t vec_end = (total / int_lanes) * int_lanes;
    for (size_t i = vec_end; i < total; ++i)
    {
        out[i] = static_cast<int32_t>(lhs[i] * alpha + rhs[i] * beta);
    }
}

void unary_negate_xsimd(const float* src, float* dst, size_t total)
{
    const xsimd::batch<int32_t> sign_i(0x80000000u);
    const Batch sign_mask = xsimd::bitwise_cast<float>(sign_i);
    const bool parallel = should_parallelize_1d_loop(total / kLanes, kLanes, 1LL << 16, 2);

    if (!parallel)
    {
        // Keep single-thread latency low for this memory-bound unary op.
        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = -src[i];
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long long i = 0; i <= static_cast<long long>(total) - static_cast<long long>(kLanes); i += static_cast<long long>(kLanes))
    {
        const auto idx = static_cast<size_t>(i);
        const Batch x = Batch::load_unaligned(src + idx);
        const Batch y = xsimd::bitwise_xor(x, sign_mask);
        y.store_unaligned(dst + idx);
    }

    const size_t vec_end = (total / kLanes) * kLanes;
    for (size_t i = vec_end; i < total; ++i)
    {
        dst[i] = -src[i];
    }
}

void unary_negate_xsimd(const int32_t* src, int32_t* dst, size_t total)
{
    using IntBatch = xsimd::batch<int32_t>;
    constexpr size_t int_lanes = IntBatch::size;
    const bool parallel = should_parallelize_1d_loop(total / int_lanes, int_lanes, 1LL << 16, 2);

    if (!parallel)
    {
        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = -src[i];
        }
        return;
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (long long i = 0; i <= static_cast<long long>(total) - static_cast<long long>(int_lanes); i += static_cast<long long>(int_lanes))
    {
        const auto idx = static_cast<size_t>(i);
        const IntBatch x = IntBatch::load_unaligned(src + idx);
        const IntBatch y = -x;
        y.store_unaligned(dst + idx);
    }

    const size_t vec_end = (total / int_lanes) * int_lanes;
    for (size_t i = vec_end; i < total; ++i)
    {
        dst[i] = -src[i];
    }
}

}  // namespace cpu
}  // namespace minfer
