#include "binary_kernel_xsimd.h"
#include "openmp_utils.h"
#include "minfer/parallel.h"

#include "xsimd/xsimd.hpp"

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
    const bool parallel = should_parallelize_1d_loop(outer, inner, 1LL << 15, 2);

    auto process_outer = [&](long long begin, long long end) {
        for (long long outer_idx = begin; outer_idx < end; ++outer_idx)
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
    };

    if (parallel)
    {
        parallel_for_1d(0, outer_ll, 1, process_outer);
    }
    else
    {
        process_outer(0, outer_ll);
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
    const long long vec_chunks = static_cast<long long>(total / kLanes);
    const bool parallel =
        should_parallelize_1d_loop(static_cast<size_t>(vec_chunks), kLanes, 1LL << 16, 2);

    auto process_chunks = [&](long long begin, long long end) {
        for (long long chunk = begin; chunk < end; ++chunk)
        {
            const size_t idx = static_cast<size_t>(chunk) * kLanes;
            const Batch a = Batch::load_unaligned(lhs + idx);
            const Batch b = Batch::load_unaligned(rhs + idx);
            const Batch y = xsimd::fma(a, alpha_batch, b * beta_batch);
            y.store_unaligned(out + idx);
        }
    };

    if (parallel)
    {
        parallel_for_1d(0, vec_chunks, 1, process_chunks);
    }
    else
    {
        process_chunks(0, vec_chunks);
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
    const long long vec_chunks = static_cast<long long>(total / int_lanes);
    const bool parallel = should_parallelize_1d_loop(
        static_cast<size_t>(vec_chunks),
        int_lanes,
        1LL << 16,
        2);

    auto process_chunks = [&](long long begin, long long end) {
        for (long long chunk = begin; chunk < end; ++chunk)
        {
            const size_t idx = static_cast<size_t>(chunk) * int_lanes;
            const IntBatch a_i = IntBatch::load_unaligned(lhs + idx);
            const IntBatch b_i = IntBatch::load_unaligned(rhs + idx);
            const FloatBatch a = xsimd::to_float(a_i);
            const FloatBatch b = xsimd::to_float(b_i);
            const FloatBatch y = xsimd::fma(a, alpha_batch, b * beta_batch);
            const IntBatch y_i = xsimd::to_int(xsimd::trunc(y));
            y_i.store_unaligned(out + idx);
        }
    };

    if (parallel)
    {
        parallel_for_1d(0, vec_chunks, 1, process_chunks);
    }
    else
    {
        process_chunks(0, vec_chunks);
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

    const long long vec_chunks = static_cast<long long>(total / kLanes);
    parallel_for_1d(0, vec_chunks, 1, [&](long long begin, long long end) {
        for (long long chunk = begin; chunk < end; ++chunk)
        {
            const size_t idx = static_cast<size_t>(chunk) * kLanes;
            const Batch x = Batch::load_unaligned(src + idx);
            const Batch y = xsimd::bitwise_xor(x, sign_mask);
            y.store_unaligned(dst + idx);
        }
    });

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

    const long long vec_chunks = static_cast<long long>(total / int_lanes);
    parallel_for_1d(0, vec_chunks, 1, [&](long long begin, long long end) {
        for (long long chunk = begin; chunk < end; ++chunk)
        {
            const size_t idx = static_cast<size_t>(chunk) * int_lanes;
            const IntBatch x = IntBatch::load_unaligned(src + idx);
            const IntBatch y = -x;
            y.store_unaligned(dst + idx);
        }
    });

    const size_t vec_end = (total / int_lanes) * int_lanes;
    for (size_t i = vec_end; i < total; ++i)
    {
        dst[i] = -src[i];
    }
}

}  // namespace cpu
}  // namespace minfer
