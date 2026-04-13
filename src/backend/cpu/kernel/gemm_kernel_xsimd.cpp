#include "gemm_kernel_xsimd.h"
#include "openmp_utils.h"
#include "xsimd_kernel_utils.h"

#include "minfer/system.h"
#include "xsimd/xsimd.hpp"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace minfer {
namespace cpu {

namespace {

constexpr int kKernelMR = 4;
constexpr int kKernelNRVecs = 2;
constexpr int kKernelNR = kKernelNRVecs * static_cast<int>(kXSimdBatchSize);

constexpr int kBlockKC = 256;
constexpr int kBlockMC = 128;
constexpr int kBlockNC = 256;

constexpr size_t kBlockedMinWork = 1ULL << 15;
constexpr long long kDecodeGemvFp16MinParallelWorkDefault = 1LL << 17;
constexpr long long kDecodeGemvFp32MinParallelWorkDefault = 1LL << 17;
constexpr long long kDecodeGemvI8MinParallelWorkDefault = 1LL << 17;

inline int ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

inline bool should_use_blocked_kernel(int m, int n, int k)
{
    if (m <= 1 || n < kKernelNR || k < static_cast<int>(kXSimdBatchSize))
    {
        return false;
    }

    const size_t work = static_cast<size_t>(std::max(m, 0)) *
                        static_cast<size_t>(std::max(n, 0)) *
                        static_cast<size_t>(std::max(k, 0));
    return work >= kBlockedMinWork;
}

inline long long parse_decode_gemv_min_parallel_work_env(const char* env_name, long long fallback)
{
    const char* env = std::getenv(env_name);
    if (!env || env[0] == '\0')
    {
        return fallback;
    }

    char* end = nullptr;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || parsed <= 0)
    {
        return fallback;
    }

    return parsed;
}

inline long long get_decode_gemv_fp16_min_parallel_work()
{
    // Env: MINFER_GEMV_FP16_MIN_PARALLEL_WORK (default: 1<<17).
    static const long long min_work = parse_decode_gemv_min_parallel_work_env(
        "MINFER_GEMV_FP16_MIN_PARALLEL_WORK",
        kDecodeGemvFp16MinParallelWorkDefault);
    return min_work;
}

inline long long get_decode_gemv_fp32_min_parallel_work()
{
    // Env: MINFER_GEMV_FP32_MIN_PARALLEL_WORK (default: 1<<17).
    static const long long min_work = parse_decode_gemv_min_parallel_work_env(
        "MINFER_GEMV_FP32_MIN_PARALLEL_WORK",
        kDecodeGemvFp32MinParallelWorkDefault);
    return min_work;
}

inline long long get_decode_gemv_i8_min_parallel_work()
{
    // Tunable threshold for decode INT8 GEMV parallelization.
    // Larger value => prefer single-thread for small workloads.
    // Env: MINFER_GEMV_I8_MIN_PARALLEL_WORK (default: 1<<17).
    static const long long min_work = parse_decode_gemv_min_parallel_work_env(
        "MINFER_GEMV_I8_MIN_PARALLEL_WORK",
        kDecodeGemvI8MinParallelWorkDefault);

    return min_work;
}

inline bool should_parallelize_decode_gemv_pairs(int pair_blocks,
                                                 int k,
                                                 int lanes_per_block,
                                                 long long min_parallel_work)
{
    if (pair_blocks <= 1 || k <= 0 || lanes_per_block <= 0)
    {
        return false;
    }

    return should_parallelize_1d_loop(
        static_cast<size_t>(pair_blocks),
        static_cast<size_t>(2) * static_cast<size_t>(k) * static_cast<size_t>(lanes_per_block),
        min_parallel_work,
        2);
}

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

inline void microkernel_4x2v(const float* packed_a,
                             const float* packed_b,
                             float* c,
                             int ldc,
                             int kc,
                             int mr,
                             int nr)
{
    XSimdBatch acc[kKernelMR][kKernelNRVecs];
    for (int r = 0; r < kKernelMR; ++r)
    {
        for (int v = 0; v < kKernelNRVecs; ++v)
        {
            acc[r][v] = XSimdBatch(0.0f);
        }
    }

    for (int p = 0; p < kc; ++p)
    {
        const float* a_ptr = packed_a + static_cast<size_t>(p) * kKernelMR;
        const float* b_ptr = packed_b + static_cast<size_t>(p) * kKernelNR;

        const XSimdBatch b0 = XSimdBatch::load_unaligned(b_ptr);
        const XSimdBatch b1 = XSimdBatch::load_unaligned(b_ptr + kXSimdBatchSize);

        acc[0][0] = xsimd::fma(XSimdBatch(a_ptr[0]), b0, acc[0][0]);
        acc[0][1] = xsimd::fma(XSimdBatch(a_ptr[0]), b1, acc[0][1]);
        acc[1][0] = xsimd::fma(XSimdBatch(a_ptr[1]), b0, acc[1][0]);
        acc[1][1] = xsimd::fma(XSimdBatch(a_ptr[1]), b1, acc[1][1]);
        acc[2][0] = xsimd::fma(XSimdBatch(a_ptr[2]), b0, acc[2][0]);
        acc[2][1] = xsimd::fma(XSimdBatch(a_ptr[2]), b1, acc[2][1]);
        acc[3][0] = xsimd::fma(XSimdBatch(a_ptr[3]), b0, acc[3][0]);
        acc[3][1] = xsimd::fma(XSimdBatch(a_ptr[3]), b1, acc[3][1]);
    }

    if (mr == kKernelMR && nr == kKernelNR)
    {
        for (int r = 0; r < kKernelMR; ++r)
        {
            float* c_row = c + static_cast<size_t>(r) * ldc;
            (XSimdBatch::load_unaligned(c_row) + acc[r][0]).store_unaligned(c_row);
            (XSimdBatch::load_unaligned(c_row + kXSimdBatchSize) + acc[r][1]).store_unaligned(c_row + kXSimdBatchSize);
        }
        return;
    }

    alignas(64) float tile[kKernelMR * kKernelNR];
    for (int r = 0; r < kKernelMR; ++r)
    {
        acc[r][0].store_unaligned(tile + r * kKernelNR);
        acc[r][1].store_unaligned(tile + r * kKernelNR + kXSimdBatchSize);
    }

    for (int r = 0; r < mr; ++r)
    {
        float* c_row = c + static_cast<size_t>(r) * ldc;
        const float* tile_row = tile + r * kKernelNR;
        for (int col = 0; col < nr; ++col)
        {
            c_row[col] += tile_row[col];
        }
    }
}

inline void pack_a_panel(const float* a,
                         int lda,
                         int row_start,
                         int col_start,
                         int mc,
                         int kc,
                         std::vector<float>& packed_a)
{
    const int row_blocks = ceil_div(mc, kKernelMR);
    packed_a.assign(static_cast<size_t>(row_blocks) * kc * kKernelMR, 0.0f);

    for (int block = 0; block < row_blocks; ++block)
    {
        float* dst_block = packed_a.data() + static_cast<size_t>(block) * kc * kKernelMR;
        const int local_row_base = block * kKernelMR;

        for (int p = 0; p < kc; ++p)
        {
            float* dst = dst_block + static_cast<size_t>(p) * kKernelMR;
            const int src_col = col_start + p;
            for (int r = 0; r < kKernelMR; ++r)
            {
                const int local_row = local_row_base + r;
                if (local_row < mc)
                {
                    dst[r] = a[static_cast<size_t>(row_start + local_row) * lda + src_col];
                }
            }
        }
    }
}

template <class LoadBScalar>
inline void pack_b_panel(int kc,
                         int nc,
                         LoadBScalar&& load_b_scalar,
                         std::vector<float>& packed_b)
{
    const int col_blocks = ceil_div(nc, kKernelNR);
    packed_b.assign(static_cast<size_t>(col_blocks) * kc * kKernelNR, 0.0f);

    for (int block = 0; block < col_blocks; ++block)
    {
        float* dst_block = packed_b.data() + static_cast<size_t>(block) * kc * kKernelNR;
        const int local_col_base = block * kKernelNR;

        for (int p = 0; p < kc; ++p)
        {
            float* dst = dst_block + static_cast<size_t>(p) * kKernelNR;
            for (int j = 0; j < kKernelNR; ++j)
            {
                const int local_col = local_col_base + j;
                if (local_col < nc)
                {
                    dst[j] = load_b_scalar(p, local_col);
                }
            }
        }
    }
}

template <class LoadBScalar>
void gemm_kernel_blocked_impl(const float* a,
                              float* c,
                              int m,
                              int n,
                              int k,
                              LoadBScalar&& load_b_scalar)
{
    std::fill_n(c, static_cast<size_t>(m) * static_cast<size_t>(n), 0.0f);

    const int jc_blocks = ceil_div(n, kBlockNC);
    const bool parallel_jc = should_parallelize_1d_loop(
        static_cast<size_t>(jc_blocks),
        static_cast<size_t>(std::max(m, 1)) * static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kBlockNC),
        1LL << 16,
        1);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_jc)
#endif
    for (int jb = 0; jb < jc_blocks; ++jb)
    {
        std::vector<float> packed_b;
        std::vector<float> packed_a;
        packed_b.reserve(static_cast<size_t>(kBlockKC) * kBlockNC);
        packed_a.reserve(static_cast<size_t>(kBlockKC) * kBlockMC);

        const int jc = jb * kBlockNC;
        const int nc = std::min(kBlockNC, n - jc);

        for (int pc = 0; pc < k; pc += kBlockKC)
        {
            const int kc = std::min(kBlockKC, k - pc);
            pack_b_panel(kc, nc,
                         [&](int p, int local_col) {
                             return load_b_scalar(pc + p, jc + local_col);
                         },
                         packed_b);

            const int col_blocks = ceil_div(nc, kKernelNR);

            for (int ic = 0; ic < m; ic += kBlockMC)
            {
                const int mc = std::min(kBlockMC, m - ic);
                pack_a_panel(a, k, ic, pc, mc, kc, packed_a);

                const int row_blocks = ceil_div(mc, kKernelMR);
                for (int block_col = 0; block_col < col_blocks; ++block_col)
                {
                    const int nr = std::min(kKernelNR, nc - block_col * kKernelNR);
                    const float* packed_b_block = packed_b.data() + static_cast<size_t>(block_col) * kc * kKernelNR;

                    for (int block_row = 0; block_row < row_blocks; ++block_row)
                    {
                        const int mr = std::min(kKernelMR, mc - block_row * kKernelMR);
                        const float* packed_a_block = packed_a.data() + static_cast<size_t>(block_row) * kc * kKernelMR;
                        float* c_block = c + static_cast<size_t>(ic + block_row * kKernelMR) * n +
                                         (jc + block_col * kKernelNR);

                        microkernel_4x2v(packed_a_block, packed_b_block, c_block, n, kc, mr, nr);
                    }
                }
            }
        }
    }
}

template <class PackedT>
inline void pack_b_raw_from_kn(const PackedT* b, PackedT* packed_b, int n, int k)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    std::fill_n(packed_b,
                static_cast<size_t>(col_blocks) * static_cast<size_t>(k) * static_cast<size_t>(kKernelNR),
                static_cast<PackedT>(0));

    for (int block = 0; block < col_blocks; ++block)
    {
        PackedT* dst_block = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        const int col_base = block * kKernelNR;
        for (int p = 0; p < k; ++p)
        {
            PackedT* dst = dst_block + static_cast<size_t>(p) * kKernelNR;
            for (int lane = 0; lane < kKernelNR; ++lane)
            {
                const int col = col_base + lane;
                if (col < n)
                {
                    dst[lane] = b[static_cast<size_t>(p) * n + col];
                }
            }
        }
    }
}

inline void pack_scales_rowwise(const float* scales, float* packed_scales, int n)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    std::fill_n(packed_scales, static_cast<size_t>(col_blocks) * kKernelNR, 0.0f);

    for (int block = 0; block < col_blocks; ++block)
    {
        float* dst = packed_scales + static_cast<size_t>(block) * kKernelNR;
        const int col_base = block * kKernelNR;
        for (int lane = 0; lane < kKernelNR; ++lane)
        {
            const int col = col_base + lane;
            if (col < n)
            {
                dst[lane] = scales[col];
            }
        }
    }
}

inline void store_row_block(float* c, int nr, const XSimdBatch& sum0, const XSimdBatch& sum1)
{
    if (nr == kKernelNR)
    {
        sum0.store_unaligned(c);
        sum1.store_unaligned(c + kXSimdBatchSize);
        return;
    }

    alignas(64) float tmp[kKernelNR];
    sum0.store_unaligned(tmp);
    sum1.store_unaligned(tmp + kXSimdBatchSize);
    for (int lane = 0; lane < nr; ++lane)
    {
        c[lane] = tmp[lane];
    }
}

template <class PackedT, class LoadPackedBatch, class ApplyBlockPostprocess>
void gemv_parallel_packed_pair_impl(const float* a,
                                    const PackedT* packed_b0,
                                    const PackedT* packed_b1,
                                    float* c0,
                                    float* c1,
                                    int n,
                                    int k,
                                    long long min_parallel_work,
                                    LoadPackedBatch&& load_packed_batch,
                                    ApplyBlockPostprocess&& apply_block_postprocess)
{
    M_Assert(a != nullptr);
    M_Assert(packed_b0 != nullptr);
    M_Assert(packed_b1 != nullptr);
    M_Assert(c0 != nullptr);
    M_Assert(c1 != nullptr);
    M_Assert(n > 0);
    M_Assert(k > 0);

    const int col_blocks = ceil_div(n, kKernelNR);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(2) * static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kKernelNR),
        min_parallel_work,
        2);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_blocks)
#endif
    for (int block = 0; block < col_blocks; ++block)
    {
        const PackedT* r0 = packed_b0 + static_cast<size_t>(block) * k * kKernelNR;
        const PackedT* r1 = packed_b1 + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch s00(0.0f), s01(0.0f);
        XSimdBatch s10(0.0f), s11(0.0f);

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s00 = xsimd::fma(av0, load_packed_batch(r0), s00);
            s01 = xsimd::fma(av0, load_packed_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av0, load_packed_batch(r1), s10);
            s11 = xsimd::fma(av0, load_packed_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s00 = xsimd::fma(av1, load_packed_batch(r0), s00);
            s01 = xsimd::fma(av1, load_packed_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av1, load_packed_batch(r1), s10);
            s11 = xsimd::fma(av1, load_packed_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s00 = xsimd::fma(av, load_packed_batch(r0), s00);
            s01 = xsimd::fma(av, load_packed_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av, load_packed_batch(r1), s10);
            s11 = xsimd::fma(av, load_packed_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }

        apply_block_postprocess(block, s00, s01, s10, s11);

        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c0 + static_cast<size_t>(block) * kKernelNR, nr, s00, s01);
        store_row_block(c1 + static_cast<size_t>(block) * kKernelNR, nr, s10, s11);
    }
}

inline bool decode_candidate_better(const DecodeCandidate& lhs, const DecodeCandidate& rhs)
{
    if (lhs.logit != rhs.logit)
    {
        return lhs.logit > rhs.logit;
    }
    return lhs.token_id < rhs.token_id;
}

inline bool decode_candidate_better(int token_id, float logit, const DecodeCandidate& rhs)
{
    if (logit != rhs.logit)
    {
        return logit > rhs.logit;
    }
    return token_id < rhs.token_id;
}

inline void update_decode_argmax_candidate(DecodeCandidate& best, int token_id, float logit)
{
    if (logit > best.logit || (logit == best.logit && (best.token_id < 0 || token_id < best.token_id)))
    {
        best.token_id = token_id;
        best.logit = logit;
    }
}

inline void update_decode_selection_candidate(DecodeSelection& selection,
                                              DecodeOutputMode mode,
                                              int top_limit,
                                              int token_id,
                                              float logit)
{
    if (mode == DecodeOutputMode::ArgMax)
    {
        DecodeCandidate best = {selection.token_id, selection.logit};
        update_decode_argmax_candidate(best, token_id, logit);
        selection.token_id = best.token_id;
        selection.logit = best.logit;
        return;
    }

    DecodeCandidate candidate = {token_id, logit};
    auto it = std::find_if(selection.top_k.begin(),
                           selection.top_k.end(),
                           [&](const DecodeCandidate& existing) { return decode_candidate_better(candidate, existing); });
    if (it != selection.top_k.end())
    {
        selection.top_k.insert(it, candidate);
    }
    else if (static_cast<int>(selection.top_k.size()) < top_limit)
    {
        selection.top_k.push_back(candidate);
    }

    if (static_cast<int>(selection.top_k.size()) > top_limit)
    {
        selection.top_k.pop_back();
    }

    if (!selection.top_k.empty())
    {
        selection.token_id = selection.top_k.front().token_id;
        selection.logit = selection.top_k.front().logit;
    }
}

inline void merge_decode_selection(DecodeSelection& dst,
                                   const DecodeSelection& src,
                                   DecodeOutputMode mode,
                                   int top_limit)
{
    if (mode == DecodeOutputMode::ArgMax)
    {
        if (src.token_id >= 0)
        {
            update_decode_selection_candidate(dst, mode, top_limit, src.token_id, src.logit);
        }
        return;
    }

    for (const DecodeCandidate& candidate : src.top_k)
    {
        update_decode_selection_candidate(dst, mode, top_limit, candidate.token_id, candidate.logit);
    }
}

inline void finalize_decode_argmax_selection(DecodeSelection& selection, const DecodeCandidate& best)
{
    selection.reset(DecodeOutputMode::ArgMax, 1);
    selection.token_id = best.token_id;
    selection.logit = best.logit;
    selection.ready = best.token_id >= 0;
}

inline std::vector<DecodeCandidate>& decode_candidate_scratch(int count)
{
    static thread_local std::vector<DecodeCandidate> scratch;
    scratch.assign(static_cast<size_t>(count), DecodeCandidate());
    return scratch;
}

inline std::vector<DecodeSelection>& decode_selection_scratch(int count)
{
    static thread_local std::vector<DecodeSelection> scratch;
    scratch.resize(static_cast<size_t>(count));
    return scratch;
}

constexpr int kFastDecodeTopKMax = 8;

template <int MaxTopK>
struct FixedDecodeTopKBuffer
{
    int limit = 0;
    int count = 0;
    std::array<DecodeCandidate, MaxTopK> items {};

    void reset(int requested_limit)
    {
        limit = std::max(1, std::min(requested_limit, MaxTopK));
        count = 0;
    }

    bool full() const
    {
        return count == limit;
    }

    const DecodeCandidate& worst() const
    {
        M_Assert(count > 0);
        return items[static_cast<size_t>(count - 1)];
    }

    void tryInsert(int token_id, float logit)
    {
        if (count == limit && !decode_candidate_better(token_id, logit, items[static_cast<size_t>(count - 1)]))
        {
            return;
        }

        int insert_pos = (count < limit) ? count : (limit - 1);
        while (insert_pos > 0 &&
               decode_candidate_better(token_id, logit, items[static_cast<size_t>(insert_pos - 1)]))
        {
            items[static_cast<size_t>(insert_pos)] = items[static_cast<size_t>(insert_pos - 1)];
            --insert_pos;
        }

        items[static_cast<size_t>(insert_pos)].token_id = token_id;
        items[static_cast<size_t>(insert_pos)].logit = logit;
        if (count < limit)
        {
            ++count;
        }
    }
};

template <int MaxTopK>
inline std::vector<FixedDecodeTopKBuffer<MaxTopK>>& decode_topk_buffer_scratch(int count)
{
    static thread_local std::vector<FixedDecodeTopKBuffer<MaxTopK>> scratch;
    scratch.resize(static_cast<size_t>(count));
    return scratch;
}

template <int MaxTopK>
inline void merge_fixed_decode_topk_buffer(FixedDecodeTopKBuffer<MaxTopK>& dst,
                                           const FixedDecodeTopKBuffer<MaxTopK>& src)
{
    for (int i = 0; i < src.count; ++i)
    {
        const DecodeCandidate& candidate = src.items[static_cast<size_t>(i)];
        dst.tryInsert(candidate.token_id, candidate.logit);
    }
}

template <int MaxTopK>
inline void finalize_decode_topk_selection(DecodeSelection& selection,
                                           const FixedDecodeTopKBuffer<MaxTopK>& buffer)
{
    selection.reset(DecodeOutputMode::TopK, buffer.limit);
    selection.top_k.assign(buffer.items.begin(), buffer.items.begin() + buffer.count);
    if (!selection.top_k.empty())
    {
        selection.token_id = selection.top_k.front().token_id;
        selection.logit = selection.top_k.front().logit;
        selection.ready = true;
    }
}

template <bool HasBias, class PackedT, class LoadPackedBatch, class ApplyBlockPostprocess>
void gemv_argmax_packed_impl(const float* a,
                             const PackedT* packed_b,
                             const float* bias,
                             int n,
                             int k,
                             DecodeSelection& selection,
                             long long min_parallel_work,
                             LoadPackedBatch&& load_packed_batch,
                             ApplyBlockPostprocess&& apply_block_postprocess)
{
    M_Assert(a != nullptr);
    M_Assert(packed_b != nullptr);
    M_Assert(n > 0);
    M_Assert(k > 0);
    M_Assert(!HasBias || bias != nullptr);

    const int col_blocks = ceil_div(n, kKernelNR);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kKernelNR),
        min_parallel_work,
        2);

    int thread_count = 1;
#ifdef _OPENMP
    if (parallel_blocks)
    {
        thread_count = omp_get_max_threads();
    }
#endif

    std::vector<DecodeCandidate>& locals = decode_candidate_scratch(thread_count);

#ifdef _OPENMP
#pragma omp parallel if(parallel_blocks)
    {
        DecodeCandidate& local = locals[static_cast<size_t>(omp_get_thread_num())];

#pragma omp for schedule(static)
        for (int block = 0; block < col_blocks; ++block)
        {
            const PackedT* r = packed_b + static_cast<size_t>(block) * k * kKernelNR;
            XSimdBatch sum0(0.0f);
            XSimdBatch sum1(0.0f);

            for (int p = 0; p < k; ++p)
            {
                const XSimdBatch av(a[p]);
                sum0 = xsimd::fma(av, load_packed_batch(r), sum0);
                sum1 = xsimd::fma(av, load_packed_batch(r + kXSimdBatchSize), sum1);
                r += kKernelNR;
            }

            apply_block_postprocess(block, sum0, sum1);

            alignas(64) float tmp[kKernelNR];
            const int nr = std::min(kKernelNR, n - block * kKernelNR);
            store_row_block(tmp, nr, sum0, sum1);

            const int token_base = block * kKernelNR;
            for (int lane = 0; lane < nr; ++lane)
            {
                const int token_id = token_base + lane;
                float logit = tmp[lane];
                if constexpr (HasBias)
                {
                    logit += bias[token_id];
                }
                update_decode_argmax_candidate(local, token_id, logit);
            }
        }
    }
#else
    DecodeCandidate& local = locals[0];
    for (int block = 0; block < col_blocks; ++block)
    {
        const PackedT* r = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            sum0 = xsimd::fma(av, load_packed_batch(r), sum0);
            sum1 = xsimd::fma(av, load_packed_batch(r + kXSimdBatchSize), sum1);
            r += kKernelNR;
        }

        apply_block_postprocess(block, sum0, sum1);

        alignas(64) float tmp[kKernelNR];
        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(tmp, nr, sum0, sum1);

        const int token_base = block * kKernelNR;
        for (int lane = 0; lane < nr; ++lane)
        {
            const int token_id = token_base + lane;
            float logit = tmp[lane];
            if constexpr (HasBias)
            {
                logit += bias[token_id];
            }
            update_decode_argmax_candidate(local, token_id, logit);
        }
    }
#endif

    DecodeCandidate best;
    for (const DecodeCandidate& local : locals)
    {
        if (local.token_id >= 0)
        {
            update_decode_argmax_candidate(best, local.token_id, local.logit);
        }
    }

    finalize_decode_argmax_selection(selection, best);
}

template <int MaxTopK, class PackedT, class LoadPackedBatch, class ApplyBlockPostprocess>
void gemv_topk_packed_impl(const float* a,
                           const PackedT* packed_b,
                           const float* bias,
                           int n,
                           int k,
                           int top_k,
                           DecodeSelection& selection,
                           long long min_parallel_work,
                           LoadPackedBatch&& load_packed_batch,
                           ApplyBlockPostprocess&& apply_block_postprocess)
{
    M_Assert(a != nullptr);
    M_Assert(packed_b != nullptr);
    M_Assert(n > 0);
    M_Assert(k > 0);

    const int top_limit = std::max(1, std::min({top_k, n, MaxTopK}));
    const int col_blocks = ceil_div(n, kKernelNR);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kKernelNR),
        min_parallel_work,
        2);

    int thread_count = 1;
#ifdef _OPENMP
    if (parallel_blocks)
    {
        thread_count = omp_get_max_threads();
    }
#endif

    std::vector<FixedDecodeTopKBuffer<MaxTopK>>& locals = decode_topk_buffer_scratch<MaxTopK>(thread_count);

#ifdef _OPENMP
#pragma omp parallel if(parallel_blocks)
    {
        FixedDecodeTopKBuffer<MaxTopK>& local = locals[static_cast<size_t>(omp_get_thread_num())];
        local.reset(top_limit);

#pragma omp for schedule(static)
        for (int block = 0; block < col_blocks; ++block)
        {
            const PackedT* r = packed_b + static_cast<size_t>(block) * k * kKernelNR;
            XSimdBatch sum0(0.0f);
            XSimdBatch sum1(0.0f);

            for (int p = 0; p < k; ++p)
            {
                const XSimdBatch av(a[p]);
                sum0 = xsimd::fma(av, load_packed_batch(r), sum0);
                sum1 = xsimd::fma(av, load_packed_batch(r + kXSimdBatchSize), sum1);
                r += kKernelNR;
            }

            apply_block_postprocess(block, sum0, sum1);

            const int nr = std::min(kKernelNR, n - block * kKernelNR);
            const int token_base = block * kKernelNR;
            const bool full_block = nr == kKernelNR;

            XSimdBatch logits0 = sum0;
            XSimdBatch logits1 = sum1;
            bool logits_include_bias = false;

            if (full_block && bias != nullptr)
            {
                const float* bias_block = bias + token_base;
                logits0 += XSimdBatch::load_unaligned(bias_block);
                logits1 += XSimdBatch::load_unaligned(bias_block + kXSimdBatchSize);
                logits_include_bias = true;
            }

            if (full_block && local.full())
            {
                const DecodeCandidate& worst = local.worst();
                const float block_max = std::max(xsimd::reduce_max(logits0), xsimd::reduce_max(logits1));
                if (block_max < worst.logit ||
                    (block_max == worst.logit && token_base >= worst.token_id))
                {
                    continue;
                }
            }

            alignas(64) float tmp[kKernelNR];
            store_row_block(tmp, nr, logits0, logits1);

            for (int lane = 0; lane < nr; ++lane)
            {
                const int token_id = token_base + lane;
                float logit = tmp[lane];
                if (!logits_include_bias && bias != nullptr)
                {
                    logit += bias[token_id];
                }
                local.tryInsert(token_id, logit);
            }
        }
    }
#else
    FixedDecodeTopKBuffer<MaxTopK>& local = locals[0];
    local.reset(top_limit);
    for (int block = 0; block < col_blocks; ++block)
    {
        const PackedT* r = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            sum0 = xsimd::fma(av, load_packed_batch(r), sum0);
            sum1 = xsimd::fma(av, load_packed_batch(r + kXSimdBatchSize), sum1);
            r += kKernelNR;
        }

        apply_block_postprocess(block, sum0, sum1);

        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        const int token_base = block * kKernelNR;
        const bool full_block = nr == kKernelNR;

        XSimdBatch logits0 = sum0;
        XSimdBatch logits1 = sum1;
        bool logits_include_bias = false;

        if (full_block && bias != nullptr)
        {
            const float* bias_block = bias + token_base;
            logits0 += XSimdBatch::load_unaligned(bias_block);
            logits1 += XSimdBatch::load_unaligned(bias_block + kXSimdBatchSize);
            logits_include_bias = true;
        }

        if (full_block && local.full())
        {
            const DecodeCandidate& worst = local.worst();
            const float block_max = std::max(xsimd::reduce_max(logits0), xsimd::reduce_max(logits1));
            if (block_max < worst.logit ||
                (block_max == worst.logit && token_base >= worst.token_id))
            {
                continue;
            }
        }

        alignas(64) float tmp[kKernelNR];
        store_row_block(tmp, nr, logits0, logits1);

        for (int lane = 0; lane < nr; ++lane)
        {
            const int token_id = token_base + lane;
            float logit = tmp[lane];
            if (!logits_include_bias && bias != nullptr)
            {
                logit += bias[token_id];
            }
            local.tryInsert(token_id, logit);
        }
    }
#endif

    FixedDecodeTopKBuffer<MaxTopK> best;
    best.reset(top_limit);
    for (const FixedDecodeTopKBuffer<MaxTopK>& local : locals)
    {
        merge_fixed_decode_topk_buffer(best, local);
    }

    finalize_decode_topk_selection(selection, best);
}

template <class PackedT, class LoadPackedBatch, class ApplyBlockPostprocess>
void gemv_select_packed_impl(const float* a,
                             const PackedT* packed_b,
                             const float* bias,
                             int n,
                             int k,
                             DecodeOutputMode mode,
                             int top_k,
                             DecodeSelection& selection,
                             long long min_parallel_work,
                             LoadPackedBatch&& load_packed_batch,
                             ApplyBlockPostprocess&& apply_block_postprocess)
{
    M_Assert(mode != DecodeOutputMode::FullLogits);
    M_Assert(a != nullptr);
    M_Assert(packed_b != nullptr);
    M_Assert(n > 0);
    M_Assert(k > 0);

    const int top_limit = mode == DecodeOutputMode::TopK ? std::max(1, std::min(top_k, n)) : 1;
    const int col_blocks = ceil_div(n, kKernelNR);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kKernelNR),
        min_parallel_work,
        2);

    int thread_count = 1;
#ifdef _OPENMP
    if (parallel_blocks)
    {
        thread_count = omp_get_max_threads();
    }
#endif

    std::vector<DecodeSelection>& locals = decode_selection_scratch(thread_count);

#ifdef _OPENMP
#pragma omp parallel if(parallel_blocks)
    {
        DecodeSelection& local = locals[static_cast<size_t>(omp_get_thread_num())];
        local.reset(mode, top_limit);

#pragma omp for schedule(static)
        for (int block = 0; block < col_blocks; ++block)
        {
            const PackedT* r = packed_b + static_cast<size_t>(block) * k * kKernelNR;
            XSimdBatch sum0(0.0f);
            XSimdBatch sum1(0.0f);

            for (int p = 0; p < k; ++p)
            {
                const XSimdBatch av(a[p]);
                sum0 = xsimd::fma(av, load_packed_batch(r), sum0);
                sum1 = xsimd::fma(av, load_packed_batch(r + kXSimdBatchSize), sum1);
                r += kKernelNR;
            }

            apply_block_postprocess(block, sum0, sum1);

            alignas(64) float tmp[kKernelNR];
            const int nr = std::min(kKernelNR, n - block * kKernelNR);
            store_row_block(tmp, nr, sum0, sum1);

            const int token_base = block * kKernelNR;
            for (int lane = 0; lane < nr; ++lane)
            {
                const int token_id = token_base + lane;
                float logit = tmp[lane];
                if (bias != nullptr)
                {
                    logit += bias[token_id];
                }
                update_decode_selection_candidate(local, mode, top_limit, token_id, logit);
            }
        }
    }
#else
    DecodeSelection& local = locals[0];
    local.reset(mode, top_limit);
    for (int block = 0; block < col_blocks; ++block)
    {
        const PackedT* r = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            sum0 = xsimd::fma(av, load_packed_batch(r), sum0);
            sum1 = xsimd::fma(av, load_packed_batch(r + kXSimdBatchSize), sum1);
            r += kKernelNR;
        }

        apply_block_postprocess(block, sum0, sum1);

        alignas(64) float tmp[kKernelNR];
        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(tmp, nr, sum0, sum1);

        const int token_base = block * kKernelNR;
        for (int lane = 0; lane < nr; ++lane)
        {
            const int token_id = token_base + lane;
            float logit = tmp[lane];
            if (bias != nullptr)
            {
                logit += bias[token_id];
            }
            update_decode_selection_candidate(local, mode, top_limit, token_id, logit);
        }
    }
#endif

    selection.reset(mode, top_limit);
    for (const DecodeSelection& local : locals)
    {
        merge_decode_selection(selection, local, mode, top_limit);
    }
    selection.ready = selection.token_id >= 0;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
__attribute__((target("avx2,f16c,fma")))
inline void store_row_block_avx8(float* c, int nr, const __m256& sum)
{
    if (nr == 8)
    {
        _mm256_storeu_ps(c, sum);
        return;
    }

    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    for (int lane = 0; lane < nr; ++lane)
    {
        c[lane] = tmp[lane];
    }
}

__attribute__((target("avx512f,avx512dq,f16c,fma")))
inline void store_row_block_avx16_split(float* c0, int nr0, float* c1, int nr1, const __m512& sum01)
{
    const __m256 s0 = _mm512_castps512_ps256(sum01);
    const __m256 s1 = _mm512_extractf32x8_ps(sum01, 1);
    store_row_block_avx8(c0, nr0, s0);
    store_row_block_avx8(c1, nr1, s1);
}

__attribute__((target("avx2,fma")))
inline void gemv_parallel_packed_fp32_avx2_impl(const float* a, const float* packed_b, float* c, int n, int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kAvxLanes, get_decode_gemv_fp32_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;

        const float* r0 = packed_b + static_cast<size_t>(block0) * k * kAvxLanes;
        const float* r1 = packed_b + static_cast<size_t>(block1) * k * kAvxLanes;
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r0), s0);
            s1 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s0 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r0), s0);
            s1 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(r0), s0);
            s1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        const int nr0 = std::min(kAvxLanes, n - block0 * kAvxLanes);
        const int nr1 = std::min(kAvxLanes, n - block1 * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block0) * kAvxLanes, nr0, s0);
        store_row_block_avx8(c + static_cast<size_t>(block1) * kAvxLanes, nr1, s1);
    }

    if (has_tail)
    {
        const int block = col_blocks - 1;
        const float* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r), s);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r), s);
            r += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av, _mm256_loadu_ps(r), s);
            r += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block) * kAvxLanes, nr, s);
    }
}

__attribute__((target("avx512f,avx512dq,f16c,fma")))
inline __m512 load_fp16_pair_as_ps_avx512(const hfloat* lo_src, const hfloat* hi_src)
{
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(lo_src)));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(hi_src)));
    const __m256i pair = _mm256_set_m128i(hi, lo);
    return _mm512_cvtph_ps(pair);
}

__attribute__((target("avx512f,avx512dq,f16c,fma")))
inline void gemv_parallel_packed_fp16_avx512_impl(const float* a, const hfloat* packed_b, float* c, int n, int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kAvxLanes, get_decode_gemv_fp16_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;

        const hfloat* r0 = packed_b + static_cast<size_t>(block0) * k * kAvxLanes;
        const hfloat* r1 = packed_b + static_cast<size_t>(block1) * k * kAvxLanes;
        __m512 s01_even = _mm512_setzero_ps();
        __m512 s01_odd = _mm512_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m512 av0 = _mm512_set1_ps(a[p]);
            s01_even = _mm512_fmadd_ps(av0, load_fp16_pair_as_ps_avx512(r0, r1), s01_even);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m512 av1 = _mm512_set1_ps(a[p + 1]);
            s01_odd = _mm512_fmadd_ps(av1, load_fp16_pair_as_ps_avx512(r0, r1), s01_odd);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }
        __m512 s01 = _mm512_add_ps(s01_even, s01_odd);
        for (; p < k; ++p)
        {
            const __m512 av = _mm512_set1_ps(a[p]);
            s01 = _mm512_fmadd_ps(av, load_fp16_pair_as_ps_avx512(r0, r1), s01);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        const int nr0 = std::min(kAvxLanes, n - block0 * kAvxLanes);
        const int nr1 = std::min(kAvxLanes, n - block1 * kAvxLanes);
        store_row_block_avx16_split(c + static_cast<size_t>(block0) * kAvxLanes,
                                    nr0,
                                    c + static_cast<size_t>(block1) * kAvxLanes,
                                    nr1,
                                    s01);
    }

    if (has_tail)
    {
        const int block = col_blocks - 1;
        const hfloat* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s_even = _mm256_setzero_ps();
        __m256 s_odd = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s_even = _mm256_fmadd_ps(av0,
                                     _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                     s_even);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s_odd = _mm256_fmadd_ps(av1,
                                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                    s_odd);
            r += kAvxLanes;
        }
        __m256 s = _mm256_add_ps(s_even, s_odd);
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))), s);
            r += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block) * kAvxLanes, nr, s);
    }
}

__attribute__((target("avx2,f16c,fma")))
inline void gemv_parallel_packed_fp16_avx2_impl(const float* a, const hfloat* packed_b, float* c, int n, int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kAvxLanes, get_decode_gemv_fp16_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;

        const hfloat* r0 = packed_b + static_cast<size_t>(block0) * k * kAvxLanes;
        const hfloat* r1 = packed_b + static_cast<size_t>(block1) * k * kAvxLanes;
        __m256 s0_even = _mm256_setzero_ps();
        __m256 s0_odd = _mm256_setzero_ps();
        __m256 s1_even = _mm256_setzero_ps();
        __m256 s1_odd = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            const __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r0))));
            const __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r1))));
            s0_even = _mm256_fmadd_ps(av0, b0, s0_even);
            s1_even = _mm256_fmadd_ps(av0, b1, s1_even);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            const __m256 b0_next = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r0))));
            const __m256 b1_next = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r1))));
            s0_odd = _mm256_fmadd_ps(av1, b0_next, s0_odd);
            s1_odd = _mm256_fmadd_ps(av1, b1_next, s1_odd);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        __m256 s0 = _mm256_add_ps(s0_even, s0_odd);
        __m256 s1 = _mm256_add_ps(s1_even, s1_odd);
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            const __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r0))));
            const __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r1))));
            s0 = _mm256_fmadd_ps(av, b0, s0);
            s1 = _mm256_fmadd_ps(av, b1, s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        const int nr0 = std::min(kAvxLanes, n - block0 * kAvxLanes);
        const int nr1 = std::min(kAvxLanes, n - block1 * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block0) * kAvxLanes, nr0, s0);
        store_row_block_avx8(c + static_cast<size_t>(block1) * kAvxLanes, nr1, s1);
    }

    if (has_tail)
    {
        const int block = col_blocks - 1;
        const hfloat* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s_even = _mm256_setzero_ps();
        __m256 s_odd = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            const __m256 b0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r))));
            s_even = _mm256_fmadd_ps(av0, b0, s_even);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            const __m256 b1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r))));
            s_odd = _mm256_fmadd_ps(av1, b1, s_odd);
            r += kAvxLanes;
        }
        __m256 s = _mm256_add_ps(s_even, s_odd);
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            const __m256 b = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r))));
            s = _mm256_fmadd_ps(av, b, s);
            r += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block) * kAvxLanes, nr, s);
    }
}

__attribute__((target("avx2,fma")))
inline void gemv_parallel_packed_pair_fp32_avx2_impl(const float* a,
                                                     const float* packed_b0,
                                                     const float* packed_b1,
                                                     float* c0,
                                                     float* c1,
                                                     int n,
                                                     int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(2) * static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kAvxLanes),
        get_decode_gemv_fp32_min_parallel_work(),
        2);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_blocks)
#endif
    for (int block = 0; block < col_blocks; ++block)
    {
        const float* r0 = packed_b0 + static_cast<size_t>(block) * k * kAvxLanes;
        const float* r1 = packed_b1 + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r0), s0);
            s1 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s0 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r0), s0);
            s1 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(r0), s0);
            s1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c0 + static_cast<size_t>(block) * kAvxLanes, nr, s0);
        store_row_block_avx8(c1 + static_cast<size_t>(block) * kAvxLanes, nr, s1);
    }
}

__attribute__((target("avx2,f16c,fma")))
inline void gemv_parallel_packed_pair_fp16_avx2_impl(const float* a,
                                                     const hfloat* packed_b0,
                                                     const hfloat* packed_b1,
                                                     float* c0,
                                                     float* c1,
                                                     int n,
                                                     int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(2) * static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kAvxLanes),
        get_decode_gemv_fp16_min_parallel_work(),
        2);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_blocks)
#endif
    for (int block = 0; block < col_blocks; ++block)
    {
        const hfloat* r0 = packed_b0 + static_cast<size_t>(block) * k * kAvxLanes;
        const hfloat* r1 = packed_b1 + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s0_even = _mm256_setzero_ps();
        __m256 s0_odd = _mm256_setzero_ps();
        __m256 s1_even = _mm256_setzero_ps();
        __m256 s1_odd = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s0_even = _mm256_fmadd_ps(
                av0,
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r0)))),
                s0_even);
            s1_even = _mm256_fmadd_ps(
                av0,
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r1)))),
                s1_even);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s0_odd = _mm256_fmadd_ps(
                av1,
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r0)))),
                s0_odd);
            s1_odd = _mm256_fmadd_ps(
                av1,
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r1)))),
                s1_odd);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }
        __m256 s0 = _mm256_add_ps(s0_even, s0_odd);
        __m256 s1 = _mm256_add_ps(s1_even, s1_odd);
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(
                av,
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r0)))),
                s0);
            s1 = _mm256_fmadd_ps(
                av,
                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r1)))),
                s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c0 + static_cast<size_t>(block) * kAvxLanes, nr, s0);
        store_row_block_avx8(c1 + static_cast<size_t>(block) * kAvxLanes, nr, s1);
    }
}

__attribute__((target("avx2,fma")))
inline __m256 load_i8x8_as_ps_avx2(const int8_t* src)
{
    const __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(static_cast<const void*>(src)));
    const __m256i v32 = _mm256_cvtepi8_epi32(v8);
    return _mm256_cvtepi32_ps(v32);
}

__attribute__((target("avx2,fma")))
inline void gemv_parallel_packed_pair_i8_rowwise_avx2_impl(const float* a,
                                                           const int8_t* packed_b0,
                                                           const float* packed_scales0,
                                                           const int8_t* packed_b1,
                                                           const float* packed_scales1,
                                                           float* c0,
                                                           float* c1,
                                                           int n,
                                                           int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(2) * static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kAvxLanes),
        get_decode_gemv_i8_min_parallel_work(),
        2);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_blocks)
#endif
    for (int block = 0; block < col_blocks; ++block)
    {
        const int8_t* r0 = packed_b0 + static_cast<size_t>(block) * k * kAvxLanes;
        const int8_t* r1 = packed_b1 + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r0), s0);
            s1 = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s0 = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r0), s0);
            s1 = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r0), s0);
            s1 = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        s0 = _mm256_mul_ps(s0, _mm256_loadu_ps(packed_scales0 + static_cast<size_t>(block) * kAvxLanes));
        s1 = _mm256_mul_ps(s1, _mm256_loadu_ps(packed_scales1 + static_cast<size_t>(block) * kAvxLanes));

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c0 + static_cast<size_t>(block) * kAvxLanes, nr, s0);
        store_row_block_avx8(c1 + static_cast<size_t>(block) * kAvxLanes, nr, s1);
    }
}

__attribute__((target("avx2,fma")))
inline void gemv_parallel_packed_i8_rowwise_avx2_impl(const float* a,
                                                       const int8_t* packed_b,
                                                       const float* packed_scales,
                                                       float* c,
                                                       int n,
                                                       int k)
{
    constexpr int kAvxLanes = 8;
    const int col_blocks = ceil_div(n, kAvxLanes);
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kAvxLanes, get_decode_gemv_i8_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;

        const int8_t* r0 = packed_b + static_cast<size_t>(block0) * k * kAvxLanes;
        const int8_t* r1 = packed_b + static_cast<size_t>(block1) * k * kAvxLanes;
        __m256 s0 = _mm256_setzero_ps();
        __m256 s1 = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r0), s0);
            s1 = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s0 = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r0), s0);
            s1 = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s0 = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r0), s0);
            s1 = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r1), s1);
            r0 += kAvxLanes;
            r1 += kAvxLanes;
        }

        const __m256 sc0 = _mm256_loadu_ps(packed_scales + static_cast<size_t>(block0) * kAvxLanes);
        const __m256 sc1 = _mm256_loadu_ps(packed_scales + static_cast<size_t>(block1) * kAvxLanes);
        s0 = _mm256_mul_ps(s0, sc0);
        s1 = _mm256_mul_ps(s1, sc1);

        const int nr0 = std::min(kAvxLanes, n - block0 * kAvxLanes);
        const int nr1 = std::min(kAvxLanes, n - block1 * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block0) * kAvxLanes, nr0, s0);
        store_row_block_avx8(c + static_cast<size_t>(block1) * kAvxLanes, nr1, s1);
    }

    if (has_tail)
    {
        const int block = col_blocks - 1;
        const int8_t* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r), s);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r), s);
            r += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r), s);
            r += kAvxLanes;
        }

        const __m256 sc = _mm256_loadu_ps(packed_scales + static_cast<size_t>(block) * kAvxLanes);
        s = _mm256_mul_ps(s, sc);

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        store_row_block_avx8(c + static_cast<size_t>(block) * kAvxLanes, nr, s);
    }
}

template <bool HasBias>
__attribute__((target("avx2,fma")))
inline void update_argmax_from_avx8(DecodeCandidate& best,
                                    int token_base,
                                    int nr,
                                    const __m256& sum,
                                    const float* bias)
{
    alignas(32) float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    for (int lane = 0; lane < nr; ++lane)
    {
        const int token_id = token_base + lane;
        float logit = tmp[lane];
        if constexpr (HasBias)
        {
            logit += bias[token_id];
        }
        update_decode_argmax_candidate(best, token_id, logit);
    }
}

template <bool HasBias>
__attribute__((target("avx2,fma")))
inline void update_argmax_state_from_avx8(__m256& best_vals,
                                          __m256i& best_ids,
                                          int token_base,
                                          const __m256& sum,
                                          const float* bias)
{
    __m256 values = sum;
    if constexpr (HasBias)
    {
        values = _mm256_add_ps(values, _mm256_loadu_ps(bias + token_base));
    }

    const __m256i lane_ids = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i candidate_ids = _mm256_add_epi32(_mm256_set1_epi32(token_base), lane_ids);
    const __m256 greater_mask = _mm256_cmp_ps(values, best_vals, _CMP_GT_OQ);
    const __m256 equal_mask = _mm256_cmp_ps(values, best_vals, _CMP_EQ_OQ);
    const __m256i lower_id_mask = _mm256_cmpgt_epi32(best_ids, candidate_ids);
    const __m256 better_mask = _mm256_or_ps(greater_mask,
                                            _mm256_and_ps(equal_mask, _mm256_castsi256_ps(lower_id_mask)));

    best_vals = _mm256_blendv_ps(best_vals, values, better_mask);
    best_ids = _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(best_ids),
                                                    _mm256_castsi256_ps(candidate_ids),
                                                    better_mask));
}

__attribute__((target("avx2,fma")))
inline void merge_argmax_state_to_candidate(DecodeCandidate& best,
                                            const __m256& best_vals,
                                            const __m256i& best_ids)
{
    alignas(32) float vals[8];
    alignas(32) int ids[8];
    _mm256_storeu_ps(vals, best_vals);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ids), best_ids);

    for (int lane = 0; lane < 8; ++lane)
    {
        if (ids[lane] >= 0 && ids[lane] != std::numeric_limits<int>::max())
        {
            update_decode_argmax_candidate(best, ids[lane], vals[lane]);
        }
    }
}

template <bool HasBias>
__attribute__((target("avx2,fma")))
inline void gemv_argmax_packed_fp32_avx2_impl(const float* a,
                                              const float* packed_b,
                                              const float* bias,
                                              int n,
                                              int k,
                                              DecodeSelection& selection)
{
    constexpr int kAvxLanes = 8;
    M_Assert(!HasBias || bias != nullptr);
    const int col_blocks = ceil_div(n, kAvxLanes);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kAvxLanes),
        get_decode_gemv_fp32_min_parallel_work(),
        2);

    int thread_count = 1;
#ifdef _OPENMP
    if (parallel_blocks)
    {
        thread_count = omp_get_max_threads();
    }
#endif

    std::vector<DecodeCandidate>& locals = decode_candidate_scratch(thread_count);

#ifdef _OPENMP
#pragma omp parallel if(parallel_blocks)
    {
        DecodeCandidate& local = locals[static_cast<size_t>(omp_get_thread_num())];
        __m256 best_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256i best_ids = _mm256_set1_epi32(std::numeric_limits<int>::max());
        bool has_vector_blocks = false;

#pragma omp for schedule(static)
        for (int block = 0; block < col_blocks; ++block)
        {
            const float* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
            __m256 s = _mm256_setzero_ps();

            int p = 0;
            for (; p + 1 < k; p += 2)
            {
                const __m256 av0 = _mm256_set1_ps(a[p]);
                s = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r), s);
                r += kAvxLanes;

                const __m256 av1 = _mm256_set1_ps(a[p + 1]);
                s = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r), s);
                r += kAvxLanes;
            }
            for (; p < k; ++p)
            {
                const __m256 av = _mm256_set1_ps(a[p]);
                s = _mm256_fmadd_ps(av, _mm256_loadu_ps(r), s);
                r += kAvxLanes;
            }

            const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
            if (nr == kAvxLanes)
            {
                update_argmax_state_from_avx8<HasBias>(best_vals, best_ids, block * kAvxLanes, s, bias);
                has_vector_blocks = true;
            }
            else
            {
                update_argmax_from_avx8<HasBias>(local, block * kAvxLanes, nr, s, bias);
            }
        }

        if (has_vector_blocks)
        {
            merge_argmax_state_to_candidate(local, best_vals, best_ids);
        }
    }
#else
    DecodeCandidate& local = locals[0];
    __m256 best_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256i best_ids = _mm256_set1_epi32(std::numeric_limits<int>::max());
    bool has_vector_blocks = false;
    for (int block = 0; block < col_blocks; ++block)
    {
        const float* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av0, _mm256_loadu_ps(r), s);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s = _mm256_fmadd_ps(av1, _mm256_loadu_ps(r), s);
            r += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av, _mm256_loadu_ps(r), s);
            r += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        if (nr == kAvxLanes)
        {
            update_argmax_state_from_avx8<HasBias>(best_vals, best_ids, block * kAvxLanes, s, bias);
            has_vector_blocks = true;
        }
        else
        {
            update_argmax_from_avx8<HasBias>(local, block * kAvxLanes, nr, s, bias);
        }
    }

    if (has_vector_blocks)
    {
        merge_argmax_state_to_candidate(local, best_vals, best_ids);
    }
#endif

    DecodeCandidate best;
    for (const DecodeCandidate& local : locals)
    {
        if (local.token_id >= 0)
        {
            update_decode_argmax_candidate(best, local.token_id, local.logit);
        }
    }

    finalize_decode_argmax_selection(selection, best);
}

template <bool HasBias>
__attribute__((target("avx2,f16c,fma")))
inline void gemv_argmax_packed_fp16_avx2_impl(const float* a,
                                              const hfloat* packed_b,
                                              const float* bias,
                                              int n,
                                              int k,
                                              DecodeSelection& selection)
{
    constexpr int kAvxLanes = 8;
    M_Assert(!HasBias || bias != nullptr);
    const int col_blocks = ceil_div(n, kAvxLanes);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kAvxLanes),
        get_decode_gemv_fp16_min_parallel_work(),
        2);

    int thread_count = 1;
#ifdef _OPENMP
    if (parallel_blocks)
    {
        thread_count = omp_get_max_threads();
    }
#endif

    std::vector<DecodeCandidate>& locals = decode_candidate_scratch(thread_count);

#ifdef _OPENMP
#pragma omp parallel if(parallel_blocks)
    {
        DecodeCandidate& local = locals[static_cast<size_t>(omp_get_thread_num())];
        __m256 best_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256i best_ids = _mm256_set1_epi32(std::numeric_limits<int>::max());
        bool has_vector_blocks = false;

#pragma omp for schedule(static)
        for (int block = 0; block < col_blocks; ++block)
        {
            const hfloat* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
            __m256 s = _mm256_setzero_ps();

            int p = 0;
            for (; p + 1 < k; p += 2)
            {
                const __m256 av0 = _mm256_set1_ps(a[p]);
                s = _mm256_fmadd_ps(av0,
                                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                    s);
                r += kAvxLanes;

                const __m256 av1 = _mm256_set1_ps(a[p + 1]);
                s = _mm256_fmadd_ps(av1,
                                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                    s);
                r += kAvxLanes;
            }
            for (; p < k; ++p)
            {
                const __m256 av = _mm256_set1_ps(a[p]);
                s = _mm256_fmadd_ps(av,
                                    _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                    s);
                r += kAvxLanes;
            }

            const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
            if (nr == kAvxLanes)
            {
                update_argmax_state_from_avx8<HasBias>(best_vals, best_ids, block * kAvxLanes, s, bias);
                has_vector_blocks = true;
            }
            else
            {
                update_argmax_from_avx8<HasBias>(local, block * kAvxLanes, nr, s, bias);
            }
        }

        if (has_vector_blocks)
        {
            merge_argmax_state_to_candidate(local, best_vals, best_ids);
        }
    }
#else
    DecodeCandidate& local = locals[0];
    __m256 best_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256i best_ids = _mm256_set1_epi32(std::numeric_limits<int>::max());
    bool has_vector_blocks = false;
    for (int block = 0; block < col_blocks; ++block)
    {
        const hfloat* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av0,
                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                s);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s = _mm256_fmadd_ps(av1,
                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                s);
            r += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av,
                                _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(r)))),
                                s);
            r += kAvxLanes;
        }

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        if (nr == kAvxLanes)
        {
            update_argmax_state_from_avx8<HasBias>(best_vals, best_ids, block * kAvxLanes, s, bias);
            has_vector_blocks = true;
        }
        else
        {
            update_argmax_from_avx8<HasBias>(local, block * kAvxLanes, nr, s, bias);
        }
    }

    if (has_vector_blocks)
    {
        merge_argmax_state_to_candidate(local, best_vals, best_ids);
    }
#endif

    DecodeCandidate best;
    for (const DecodeCandidate& local : locals)
    {
        if (local.token_id >= 0)
        {
            update_decode_argmax_candidate(best, local.token_id, local.logit);
        }
    }

    finalize_decode_argmax_selection(selection, best);
}

template <bool HasBias>
__attribute__((target("avx2,fma")))
inline void gemv_argmax_packed_i8_rowwise_avx2_impl(const float* a,
                                                    const int8_t* packed_b,
                                                    const float* packed_scales,
                                                    const float* bias,
                                                    int n,
                                                    int k,
                                                    DecodeSelection& selection)
{
    constexpr int kAvxLanes = 8;
    M_Assert(packed_scales != nullptr);
    M_Assert(!HasBias || bias != nullptr);

    const int col_blocks = ceil_div(n, kAvxLanes);
    const bool parallel_blocks = should_parallelize_1d_loop(
        static_cast<size_t>(col_blocks),
        static_cast<size_t>(std::max(k, 1)) * static_cast<size_t>(kAvxLanes),
        get_decode_gemv_i8_min_parallel_work(),
        2);

    int thread_count = 1;
#ifdef _OPENMP
    if (parallel_blocks)
    {
        thread_count = omp_get_max_threads();
    }
#endif

    std::vector<DecodeCandidate>& locals = decode_candidate_scratch(thread_count);

#ifdef _OPENMP
#pragma omp parallel if(parallel_blocks)
    {
        DecodeCandidate& local = locals[static_cast<size_t>(omp_get_thread_num())];
        __m256 best_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        __m256i best_ids = _mm256_set1_epi32(std::numeric_limits<int>::max());
        bool has_vector_blocks = false;

#pragma omp for schedule(static)
        for (int block = 0; block < col_blocks; ++block)
        {
            const int8_t* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
            __m256 s = _mm256_setzero_ps();

            int p = 0;
            for (; p + 1 < k; p += 2)
            {
                const __m256 av0 = _mm256_set1_ps(a[p]);
                s = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r), s);
                r += kAvxLanes;

                const __m256 av1 = _mm256_set1_ps(a[p + 1]);
                s = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r), s);
                r += kAvxLanes;
            }
            for (; p < k; ++p)
            {
                const __m256 av = _mm256_set1_ps(a[p]);
                s = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r), s);
                r += kAvxLanes;
            }

            const __m256 sc = _mm256_loadu_ps(packed_scales + static_cast<size_t>(block) * kAvxLanes);
            s = _mm256_mul_ps(s, sc);

            const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
            if (nr == kAvxLanes)
            {
                update_argmax_state_from_avx8<HasBias>(best_vals, best_ids, block * kAvxLanes, s, bias);
                has_vector_blocks = true;
            }
            else
            {
                update_argmax_from_avx8<HasBias>(local, block * kAvxLanes, nr, s, bias);
            }
        }

        if (has_vector_blocks)
        {
            merge_argmax_state_to_candidate(local, best_vals, best_ids);
        }
    }
#else
    DecodeCandidate& local = locals[0];
    __m256 best_vals = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    __m256i best_ids = _mm256_set1_epi32(std::numeric_limits<int>::max());
    bool has_vector_blocks = false;
    for (int block = 0; block < col_blocks; ++block)
    {
        const int8_t* r = packed_b + static_cast<size_t>(block) * k * kAvxLanes;
        __m256 s = _mm256_setzero_ps();

        int p = 0;
        for (; p + 1 < k; p += 2)
        {
            const __m256 av0 = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av0, load_i8x8_as_ps_avx2(r), s);
            r += kAvxLanes;

            const __m256 av1 = _mm256_set1_ps(a[p + 1]);
            s = _mm256_fmadd_ps(av1, load_i8x8_as_ps_avx2(r), s);
            r += kAvxLanes;
        }
        for (; p < k; ++p)
        {
            const __m256 av = _mm256_set1_ps(a[p]);
            s = _mm256_fmadd_ps(av, load_i8x8_as_ps_avx2(r), s);
            r += kAvxLanes;
        }

        const __m256 sc = _mm256_loadu_ps(packed_scales + static_cast<size_t>(block) * kAvxLanes);
        s = _mm256_mul_ps(s, sc);

        const int nr = std::min(kAvxLanes, n - block * kAvxLanes);
        if (nr == kAvxLanes)
        {
            update_argmax_state_from_avx8<HasBias>(best_vals, best_ids, block * kAvxLanes, s, bias);
            has_vector_blocks = true;
        }
        else
        {
            update_argmax_from_avx8<HasBias>(local, block * kAvxLanes, nr, s, bias);
        }
    }

    if (has_vector_blocks)
    {
        merge_argmax_state_to_candidate(local, best_vals, best_ids);
    }
#endif

    DecodeCandidate best;
    for (const DecodeCandidate& local : locals)
    {
        if (local.token_id >= 0)
        {
            update_decode_argmax_candidate(best, local.token_id, local.logit);
        }
    }

    finalize_decode_argmax_selection(selection, best);
}
#endif

template <class PackedT, class LoadPackedVec>
inline void gemm_kernel_xsimd_row_packed_impl(const float* a,
                                              const PackedT* packed_b,
                                              float* c,
                                              int n,
                                              int k,
                                              LoadPackedVec&& load_packed_vec)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    for (int block = 0; block < col_blocks; ++block)
    {
        const PackedT* block_ptr = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            XSimdBatch b0(0.0f);
            XSimdBatch b1(0.0f);
            load_packed_vec(block_ptr + static_cast<size_t>(p) * kKernelNR, b0, b1);

            const XSimdBatch a_vec(a[p]);
            sum0 = xsimd::fma(a_vec, b0, sum0);
            sum1 = xsimd::fma(a_vec, b1, sum1);
        }

        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr, sum0, sum1);
    }
}

inline void gemm_kernel_xsimd_row_packed_i8_impl(const float* a,
                                                 const int8_t* packed_b,
                                                 const float* packed_scales,
                                                 float* c,
                                                 int n,
                                                 int k)
{
    const int col_blocks = ceil_div(n, kKernelNR);
    for (int block = 0; block < col_blocks; ++block)
    {
        const int8_t* block_ptr = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        const float* scale_ptr = packed_scales + static_cast<size_t>(block) * kKernelNR;
        XSimdBatch sum0(0.0f);
        XSimdBatch sum1(0.0f);

        for (int p = 0; p < k; ++p)
        {
            const int8_t* b_ptr = block_ptr + static_cast<size_t>(p) * kKernelNR;
            const XSimdBatch a_vec(a[p]);
            const XSimdBatch b0 = load_int8_batch(b_ptr);
            const XSimdBatch b1 = load_int8_batch(b_ptr + kXSimdBatchSize);
            sum0 = xsimd::fma(a_vec, b0, sum0);
            sum1 = xsimd::fma(a_vec, b1, sum1);
        }

        const XSimdBatch scale0 = XSimdBatch::load_unaligned(scale_ptr);
        const XSimdBatch scale1 = XSimdBatch::load_unaligned(scale_ptr + kXSimdBatchSize);
        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr, sum0 * scale0, sum1 * scale1);
    }
}

void gemm_kernel_xsimd_nn_simple_fp32(const float* a, const float* b, float* c,
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

void gemm_kernel_xsimd_nt_simple_fp32(const float* a, const float* b, float* c,
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

void gemm_kernel_xsimd_nn_simple_fp16(const float* a, const hfloat* b, float* c,
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

void gemm_kernel_xsimd_nt_simple_fp16(const float* a, const hfloat* b, float* c,
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

void gemm_kernel_xsimd_nn_simple_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
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
            const XSimdBatch scale_vec = XSimdBatch::load_unaligned(scales + ni);
            for (int ki = 0; ki < k; ++ki)
            {
                const XSimdBatch a_vec(a_row[ki]);
                const XSimdBatch b_vec = load_int8_batch(b + static_cast<size_t>(ki) * n + ni);
                sum_vec = xsimd::fma(a_vec, b_vec, sum_vec);
            }
            (sum_vec * scale_vec).store_unaligned(c_row + ni);
        }

        for (; ni < n; ++ni)
        {
            float sum = 0.0f;
            for (int ki = 0; ki < k; ++ki)
            {
                sum += a_row[ki] * static_cast<float>(b[static_cast<size_t>(ki) * n + ni]);
            }
            c_row[ni] = sum * scales[ni];
        }
    }
}

void gemm_kernel_xsimd_nt_simple_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
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

}  // namespace

std::size_t gemm_xsimd_packed_b_elements(int n, int k)
{
    return static_cast<std::size_t>(ceil_div(n, kKernelNR)) *
           static_cast<std::size_t>(std::max(k, 0)) *
           static_cast<std::size_t>(kKernelNR);
}

std::size_t gemm_xsimd_packed_scale_elements(int n)
{
    return static_cast<std::size_t>(ceil_div(n, kKernelNR)) *
           static_cast<std::size_t>(kKernelNR);
}

void gemm_pack_xsimd_nn_fp32(const float* b, float* packed_b, int n, int k)
{
    pack_b_raw_from_kn(b, packed_b, n, k);
}

void gemm_pack_xsimd_nn_fp16(const hfloat* b, hfloat* packed_b, int n, int k)
{
    pack_b_raw_from_kn(b, packed_b, n, k);
}

void gemm_pack_xsimd_nn_i8_rowwise(const int8_t* b, const float* scales, int8_t* packed_b, float* packed_scales,
                                   int n, int k)
{
    pack_b_raw_from_kn(b, packed_b, n, k);
    pack_scales_rowwise(scales, packed_scales, n);
}

void gemm_kernel_xsimd_row_packed_fp32(const float* a, const float* packed_b, float* c, int n, int k)
{
    gemm_kernel_xsimd_row_packed_impl(
        a, packed_b, c, n, k,
        [](const float* src, XSimdBatch& b0, XSimdBatch& b1) {
            b0 = XSimdBatch::load_unaligned(src);
            b1 = XSimdBatch::load_unaligned(src + kXSimdBatchSize);
        });
}

void gemm_kernel_xsimd_row_packed_fp16(const float* a, const hfloat* packed_b, float* c, int n, int k)
{
    gemm_kernel_xsimd_row_packed_impl(
        a, packed_b, c, n, k,
        [](const hfloat* src, XSimdBatch& b0, XSimdBatch& b1) {
            b0 = load_hfloat_batch(src);
            b1 = load_hfloat_batch(src + kXSimdBatchSize);
        });
}

void gemm_kernel_xsimd_row_packed_i8_rowwise(const float* a, const int8_t* packed_b, const float* packed_scales,
                                             float* c, int n, int k)
{
    gemm_kernel_xsimd_row_packed_i8_impl(a, packed_b, packed_scales, c, n, k);
}

void gemm_kernel_xsimd_nn(const float* a, const float* b, float* c,
                          int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nn_simple_fp32(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return b[static_cast<size_t>(src_k) * n + src_n];
        });
}

void gemm_kernel_xsimd_nt(const float* a, const float* b, float* c,
                          int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nt_simple_fp32(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return b[static_cast<size_t>(src_n) * k + src_k];
        });
}

void gemm_kernel_xsimd_nn_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nn_simple_fp16(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_k) * n + src_n]);
        });
}

void gemm_kernel_xsimd_nn_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nn_simple_i8_rowwise(a, b, scales, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_k) * n + src_n]) * scales[src_n];
        });
}

void gemm_kernel_xsimd_nt_fp16(const float* a, const hfloat* b, float* c,
                               int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nt_simple_fp16(a, b, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_n) * k + src_k]);
        });
}

void gemm_kernel_xsimd_nt_i8_rowwise(const float* a, const int8_t* b, const float* scales, float* c,
                                     int m, int n, int k)
{
    if (!should_use_blocked_kernel(m, n, k))
    {
        gemm_kernel_xsimd_nt_simple_i8_rowwise(a, b, scales, c, m, n, k);
        return;
    }

    gemm_kernel_blocked_impl(
        a, c, m, n, k,
        [&](int src_k, int src_n) {
            return static_cast<float>(b[static_cast<size_t>(src_n) * k + src_k]) * scales[src_n];
        });
}

// ═══════════════════════════════════════════════════════════════════════════
// Parallel GEMV micro-kernels: M=1, OMP-parallelized across N (column blocks)
//
// Problem:  For vocab projection (e.g. [1,512] × [32000,512]^T), the old code
//           did OMP over M=1 rows → 1 thread active, 3 idle.
// Solution: Distribute the ~2000 column-blocks across all available threads.
//           Each block: broadcast a[p], load NR packed B values, FMA accumulate.
//           2-block unrolling → 4 accumulators → better FMA pipeline utilization.
// ═══════════════════════════════════════════════════════════════════════════

void gemv_parallel_packed_fp32(const float* a, const float* packed_b, float* c, int n, int k)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_fp32_avx2_impl(a, packed_b, c, n, k);
        return;
    }
#endif

    const int col_blocks = ceil_div(n, kKernelNR);

    // Process pairs of column blocks for better ILP (4 FMA accumulators instead of 2)
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kKernelNR, get_decode_gemv_fp32_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;
        const float* bp0 = packed_b + static_cast<size_t>(block0) * k * kKernelNR;
        const float* bp1 = packed_b + static_cast<size_t>(block1) * k * kKernelNR;

        XSimdBatch s00(0.0f), s01(0.0f);  // block0 accumulators
        XSimdBatch s10(0.0f), s11(0.0f);  // block1 accumulators

        int p = 0;
        const float* r0 = bp0;
        const float* r1 = bp1;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s00 = xsimd::fma(av0, XSimdBatch::load_unaligned(r0), s00);
            s01 = xsimd::fma(av0, XSimdBatch::load_unaligned(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av0, XSimdBatch::load_unaligned(r1), s10);
            s11 = xsimd::fma(av0, XSimdBatch::load_unaligned(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s00 = xsimd::fma(av1, XSimdBatch::load_unaligned(r0), s00);
            s01 = xsimd::fma(av1, XSimdBatch::load_unaligned(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av1, XSimdBatch::load_unaligned(r1), s10);
            s11 = xsimd::fma(av1, XSimdBatch::load_unaligned(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s00 = xsimd::fma(av, XSimdBatch::load_unaligned(r0), s00);
            s01 = xsimd::fma(av, XSimdBatch::load_unaligned(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av, XSimdBatch::load_unaligned(r1), s10);
            s11 = xsimd::fma(av, XSimdBatch::load_unaligned(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }

        const int nr0 = std::min(kKernelNR, n - block0 * kKernelNR);
        store_row_block(c + static_cast<size_t>(block0) * kKernelNR, nr0, s00, s01);
        const int nr1 = std::min(kKernelNR, n - block1 * kKernelNR);
        store_row_block(c + static_cast<size_t>(block1) * kKernelNR, nr1, s10, s11);
    }

    // Handle odd tail block (at most 1, always single-threaded)
    if (has_tail)
    {
        const int block = col_blocks - 1;
        const float* bp = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch s0(0.0f), s1(0.0f);

        int p = 0;
        const float* r = bp;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s0 = xsimd::fma(av0, XSimdBatch::load_unaligned(r), s0);
            s1 = xsimd::fma(av0, XSimdBatch::load_unaligned(r + kXSimdBatchSize), s1);
            r += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s0 = xsimd::fma(av1, XSimdBatch::load_unaligned(r), s0);
            s1 = xsimd::fma(av1, XSimdBatch::load_unaligned(r + kXSimdBatchSize), s1);
            r += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s0 = xsimd::fma(av, XSimdBatch::load_unaligned(r), s0);
            s1 = xsimd::fma(av, XSimdBatch::load_unaligned(r + kXSimdBatchSize), s1);
            r += kKernelNR;
        }

        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr, s0, s1);
    }
}

void gemv_parallel_packed_fp16(const float* a, const hfloat* packed_b, float* c, int n, int k)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_fp16_avx512_impl(a, packed_b, c, n, k);
        return;
    }

    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_fp16_avx2_impl(a, packed_b, c, n, k);
        return;
    }
#endif

    const int col_blocks = ceil_div(n, kKernelNR);
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kKernelNR, get_decode_gemv_fp16_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;
        const hfloat* bp0 = packed_b + static_cast<size_t>(block0) * k * kKernelNR;
        const hfloat* bp1 = packed_b + static_cast<size_t>(block1) * k * kKernelNR;

        XSimdBatch s00(0.0f), s01(0.0f);
        XSimdBatch s10(0.0f), s11(0.0f);

        int p = 0;
        const hfloat* r0 = bp0;
        const hfloat* r1 = bp1;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s00 = xsimd::fma(av0, load_hfloat_batch(r0), s00);
            s01 = xsimd::fma(av0, load_hfloat_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av0, load_hfloat_batch(r1), s10);
            s11 = xsimd::fma(av0, load_hfloat_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s00 = xsimd::fma(av1, load_hfloat_batch(r0), s00);
            s01 = xsimd::fma(av1, load_hfloat_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av1, load_hfloat_batch(r1), s10);
            s11 = xsimd::fma(av1, load_hfloat_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s00 = xsimd::fma(av, load_hfloat_batch(r0), s00);
            s01 = xsimd::fma(av, load_hfloat_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av, load_hfloat_batch(r1), s10);
            s11 = xsimd::fma(av, load_hfloat_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }

        const int nr0 = std::min(kKernelNR, n - block0 * kKernelNR);
        store_row_block(c + static_cast<size_t>(block0) * kKernelNR, nr0, s00, s01);
        const int nr1 = std::min(kKernelNR, n - block1 * kKernelNR);
        store_row_block(c + static_cast<size_t>(block1) * kKernelNR, nr1, s10, s11);
    }

    if (has_tail)
    {
        const int block = col_blocks - 1;
        const hfloat* bp = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch s0(0.0f), s1(0.0f);

        int p = 0;
        const hfloat* r = bp;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s0 = xsimd::fma(av0, load_hfloat_batch(r), s0);
            s1 = xsimd::fma(av0, load_hfloat_batch(r + kXSimdBatchSize), s1);
            r += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s0 = xsimd::fma(av1, load_hfloat_batch(r), s0);
            s1 = xsimd::fma(av1, load_hfloat_batch(r + kXSimdBatchSize), s1);
            r += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s0 = xsimd::fma(av, load_hfloat_batch(r), s0);
            s1 = xsimd::fma(av, load_hfloat_batch(r + kXSimdBatchSize), s1);
            r += kKernelNR;
        }

        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr, s0, s1);
    }
}

void gemv_parallel_packed_i8_rowwise(const float* a, const int8_t* packed_b, const float* packed_scales,
                                     float* c, int n, int k)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_i8_rowwise_avx2_impl(a, packed_b, packed_scales, c, n, k);
        return;
    }
#endif

    const int col_blocks = ceil_div(n, kKernelNR);
    const int paired_blocks = col_blocks / 2;
    const bool has_tail = (col_blocks & 1) != 0;
    const bool parallel_pairs = should_parallelize_decode_gemv_pairs(
        paired_blocks, k, kKernelNR, get_decode_gemv_i8_min_parallel_work());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(parallel_pairs)
#endif
    for (int pair = 0; pair < paired_blocks; ++pair)
    {
        const int block0 = pair * 2;
        const int block1 = block0 + 1;
        const int8_t* bp0 = packed_b + static_cast<size_t>(block0) * k * kKernelNR;
        const int8_t* bp1 = packed_b + static_cast<size_t>(block1) * k * kKernelNR;

        XSimdBatch s00(0.0f), s01(0.0f);
        XSimdBatch s10(0.0f), s11(0.0f);

        int p = 0;
        const int8_t* r0 = bp0;
        const int8_t* r1 = bp1;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s00 = xsimd::fma(av0, load_int8_batch(r0), s00);
            s01 = xsimd::fma(av0, load_int8_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av0, load_int8_batch(r1), s10);
            s11 = xsimd::fma(av0, load_int8_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s00 = xsimd::fma(av1, load_int8_batch(r0), s00);
            s01 = xsimd::fma(av1, load_int8_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av1, load_int8_batch(r1), s10);
            s11 = xsimd::fma(av1, load_int8_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s00 = xsimd::fma(av, load_int8_batch(r0), s00);
            s01 = xsimd::fma(av, load_int8_batch(r0 + kXSimdBatchSize), s01);
            s10 = xsimd::fma(av, load_int8_batch(r1), s10);
            s11 = xsimd::fma(av, load_int8_batch(r1 + kXSimdBatchSize), s11);
            r0 += kKernelNR;
            r1 += kKernelNR;
        }

        const float* sc0 = packed_scales + static_cast<size_t>(block0) * kKernelNR;
        const float* sc1 = packed_scales + static_cast<size_t>(block1) * kKernelNR;
        const int nr0 = std::min(kKernelNR, n - block0 * kKernelNR);
        const int nr1 = std::min(kKernelNR, n - block1 * kKernelNR);
        store_row_block(c + static_cast<size_t>(block0) * kKernelNR, nr0,
                        s00 * XSimdBatch::load_unaligned(sc0),
                        s01 * XSimdBatch::load_unaligned(sc0 + kXSimdBatchSize));
        store_row_block(c + static_cast<size_t>(block1) * kKernelNR, nr1,
                        s10 * XSimdBatch::load_unaligned(sc1),
                        s11 * XSimdBatch::load_unaligned(sc1 + kXSimdBatchSize));
    }

    if (has_tail)
    {
        const int block = col_blocks - 1;
        const int8_t* bp = packed_b + static_cast<size_t>(block) * k * kKernelNR;
        XSimdBatch s0(0.0f), s1(0.0f);

        int p = 0;
        const int8_t* r = bp;
        for (; p + 1 < k; p += 2)
        {
            const XSimdBatch av0(a[p]);
            s0 = xsimd::fma(av0, load_int8_batch(r), s0);
            s1 = xsimd::fma(av0, load_int8_batch(r + kXSimdBatchSize), s1);
            r += kKernelNR;

            const XSimdBatch av1(a[p + 1]);
            s0 = xsimd::fma(av1, load_int8_batch(r), s0);
            s1 = xsimd::fma(av1, load_int8_batch(r + kXSimdBatchSize), s1);
            r += kKernelNR;
        }
        for (; p < k; ++p)
        {
            const XSimdBatch av(a[p]);
            s0 = xsimd::fma(av, load_int8_batch(r), s0);
            s1 = xsimd::fma(av, load_int8_batch(r + kXSimdBatchSize), s1);
            r += kKernelNR;
        }

        const float* sc = packed_scales + static_cast<size_t>(block) * kKernelNR;
        const int nr = std::min(kKernelNR, n - block * kKernelNR);
        store_row_block(c + static_cast<size_t>(block) * kKernelNR, nr,
                        s0 * XSimdBatch::load_unaligned(sc),
                        s1 * XSimdBatch::load_unaligned(sc + kXSimdBatchSize));
    }
}

void gemv_parallel_packed_pair_fp32(const float* a,
                                    const float* packed_b0,
                                    const float* packed_b1,
                                    float* c0,
                                    float* c1,
                                    int n,
                                    int k)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_pair_fp32_avx2_impl(a, packed_b0, packed_b1, c0, c1, n, k);
        return;
    }
#endif

    gemv_parallel_packed_pair_impl(
        a,
        packed_b0,
        packed_b1,
        c0,
        c1,
        n,
        k,
        get_decode_gemv_fp32_min_parallel_work(),
        [](const float* src) { return XSimdBatch::load_unaligned(src); },
        [](int, XSimdBatch&, XSimdBatch&, XSimdBatch&, XSimdBatch&) {});
}

void gemv_parallel_packed_pair_fp16(const float* a,
                                    const hfloat* packed_b0,
                                    const hfloat* packed_b1,
                                    float* c0,
                                    float* c1,
                                    int n,
                                    int k)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("f16c") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_pair_fp16_avx2_impl(a, packed_b0, packed_b1, c0, c1, n, k);
        return;
    }
#endif

    gemv_parallel_packed_pair_impl(
        a,
        packed_b0,
        packed_b1,
        c0,
        c1,
        n,
        k,
        get_decode_gemv_fp16_min_parallel_work(),
        [](const hfloat* src) { return load_hfloat_batch(src); },
        [](int, XSimdBatch&, XSimdBatch&, XSimdBatch&, XSimdBatch&) {});
}

void gemv_parallel_packed_pair_i8_rowwise(const float* a,
                                          const int8_t* packed_b0,
                                          const float* packed_scales0,
                                          const int8_t* packed_b1,
                                          const float* packed_scales1,
                                          float* c0,
                                          float* c1,
                                          int n,
                                          int k)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if (kKernelNR == 8 &&
        __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma"))
    {
        gemv_parallel_packed_pair_i8_rowwise_avx2_impl(
            a, packed_b0, packed_scales0, packed_b1, packed_scales1, c0, c1, n, k);
        return;
    }
#endif

    gemv_parallel_packed_pair_impl(
        a,
        packed_b0,
        packed_b1,
        c0,
        c1,
        n,
        k,
        get_decode_gemv_i8_min_parallel_work(),
        [](const int8_t* src) { return load_int8_batch(src); },
        [&](int block, XSimdBatch& s00, XSimdBatch& s01, XSimdBatch& s10, XSimdBatch& s11) {
            const float* scale0 = packed_scales0 + static_cast<size_t>(block) * kKernelNR;
            const float* scale1 = packed_scales1 + static_cast<size_t>(block) * kKernelNR;
            s00 *= XSimdBatch::load_unaligned(scale0);
            s01 *= XSimdBatch::load_unaligned(scale0 + kXSimdBatchSize);
            s10 *= XSimdBatch::load_unaligned(scale1);
            s11 *= XSimdBatch::load_unaligned(scale1 + kXSimdBatchSize);
        });
}

void gemv_select_packed_fp32(const float* a,
                             const float* packed_b,
                             const float* bias,
                             int n,
                             int k,
                             DecodeOutputMode mode,
                             int top_k,
                             DecodeSelection& selection)
{
    if (mode == DecodeOutputMode::ArgMax)
    {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
        if (kKernelNR == 8 &&
            __builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("fma"))
        {
            if (bias != nullptr)
            {
                gemv_argmax_packed_fp32_avx2_impl<true>(a, packed_b, bias, n, k, selection);
            }
            else
            {
                gemv_argmax_packed_fp32_avx2_impl<false>(a, packed_b, nullptr, n, k, selection);
            }
            return;
        }
#endif
        if (bias != nullptr)
        {
            gemv_argmax_packed_impl<true>(a,
                                          packed_b,
                                          bias,
                                          n,
                                          k,
                                          selection,
                                          get_decode_gemv_fp32_min_parallel_work(),
                                          [](const float* src) { return XSimdBatch::load_unaligned(src); },
                                          [](int, XSimdBatch&, XSimdBatch&) {});
        }
        else
        {
            gemv_argmax_packed_impl<false>(a,
                                           packed_b,
                                           nullptr,
                                           n,
                                           k,
                                           selection,
                                           get_decode_gemv_fp32_min_parallel_work(),
                                           [](const float* src) { return XSimdBatch::load_unaligned(src); },
                                           [](int, XSimdBatch&, XSimdBatch&) {});
        }
        return;
    }

    if (mode == DecodeOutputMode::TopK && top_k <= kFastDecodeTopKMax)
    {
        gemv_topk_packed_impl<kFastDecodeTopKMax>(a,
                                                  packed_b,
                                                  bias,
                                                  n,
                                                  k,
                                                  top_k,
                                                  selection,
                                                  get_decode_gemv_fp32_min_parallel_work(),
                                                  [](const float* src) { return XSimdBatch::load_unaligned(src); },
                                                  [](int, XSimdBatch&, XSimdBatch&) {});
        return;
    }

    gemv_select_packed_impl(a,
                            packed_b,
                            bias,
                            n,
                            k,
                            mode,
                            top_k,
                            selection,
                            get_decode_gemv_fp32_min_parallel_work(),
                            [](const float* src) { return XSimdBatch::load_unaligned(src); },
                            [](int, XSimdBatch&, XSimdBatch&) {});
}

void gemv_select_packed_fp16(const float* a,
                             const hfloat* packed_b,
                             const float* bias,
                             int n,
                             int k,
                             DecodeOutputMode mode,
                             int top_k,
                             DecodeSelection& selection)
{
    if (mode == DecodeOutputMode::ArgMax)
    {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
        if (kKernelNR == 8 &&
            __builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("f16c") &&
            __builtin_cpu_supports("fma"))
        {
            if (bias != nullptr)
            {
                gemv_argmax_packed_fp16_avx2_impl<true>(a, packed_b, bias, n, k, selection);
            }
            else
            {
                gemv_argmax_packed_fp16_avx2_impl<false>(a, packed_b, nullptr, n, k, selection);
            }
            return;
        }
#endif
        if (bias != nullptr)
        {
            gemv_argmax_packed_impl<true>(a,
                                          packed_b,
                                          bias,
                                          n,
                                          k,
                                          selection,
                                          get_decode_gemv_fp16_min_parallel_work(),
                                          [](const hfloat* src) { return load_hfloat_batch(src); },
                                          [](int, XSimdBatch&, XSimdBatch&) {});
        }
        else
        {
            gemv_argmax_packed_impl<false>(a,
                                           packed_b,
                                           nullptr,
                                           n,
                                           k,
                                           selection,
                                           get_decode_gemv_fp16_min_parallel_work(),
                                           [](const hfloat* src) { return load_hfloat_batch(src); },
                                           [](int, XSimdBatch&, XSimdBatch&) {});
        }
        return;
    }

    if (mode == DecodeOutputMode::TopK && top_k <= kFastDecodeTopKMax)
    {
        gemv_topk_packed_impl<kFastDecodeTopKMax>(a,
                                                  packed_b,
                                                  bias,
                                                  n,
                                                  k,
                                                  top_k,
                                                  selection,
                                                  get_decode_gemv_fp16_min_parallel_work(),
                                                  [](const hfloat* src) { return load_hfloat_batch(src); },
                                                  [](int, XSimdBatch&, XSimdBatch&) {});
        return;
    }

    gemv_select_packed_impl(a,
                            packed_b,
                            bias,
                            n,
                            k,
                            mode,
                            top_k,
                            selection,
                            get_decode_gemv_fp16_min_parallel_work(),
                            [](const hfloat* src) { return load_hfloat_batch(src); },
                            [](int, XSimdBatch&, XSimdBatch&) {});
}

void gemv_select_packed_i8_rowwise(const float* a,
                                   const int8_t* packed_b,
                                   const float* packed_scales,
                                   const float* bias,
                                   int n,
                                   int k,
                                   DecodeOutputMode mode,
                                   int top_k,
                                   DecodeSelection& selection)
{
    M_Assert(packed_scales != nullptr);

    if (mode == DecodeOutputMode::ArgMax)
    {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
        if (kKernelNR == 8 &&
            __builtin_cpu_supports("avx2") &&
            __builtin_cpu_supports("fma"))
        {
            if (bias != nullptr)
            {
                gemv_argmax_packed_i8_rowwise_avx2_impl<true>(a, packed_b, packed_scales, bias, n, k, selection);
            }
            else
            {
                gemv_argmax_packed_i8_rowwise_avx2_impl<false>(a, packed_b, packed_scales, nullptr, n, k, selection);
            }
            return;
        }
#endif
        if (bias != nullptr)
        {
            gemv_argmax_packed_impl<true>(a,
                                          packed_b,
                                          bias,
                                          n,
                                          k,
                                          selection,
                                          get_decode_gemv_i8_min_parallel_work(),
                                          [](const int8_t* src) { return load_int8_batch(src); },
                                          [&](int block, XSimdBatch& sum0, XSimdBatch& sum1) {
                                              const float* scale = packed_scales + static_cast<size_t>(block) * kKernelNR;
                                              sum0 *= XSimdBatch::load_unaligned(scale);
                                              sum1 *= XSimdBatch::load_unaligned(scale + kXSimdBatchSize);
                                          });
        }
        else
        {
            gemv_argmax_packed_impl<false>(a,
                                           packed_b,
                                           nullptr,
                                           n,
                                           k,
                                           selection,
                                           get_decode_gemv_i8_min_parallel_work(),
                                           [](const int8_t* src) { return load_int8_batch(src); },
                                           [&](int block, XSimdBatch& sum0, XSimdBatch& sum1) {
                                               const float* scale = packed_scales + static_cast<size_t>(block) * kKernelNR;
                                               sum0 *= XSimdBatch::load_unaligned(scale);
                                               sum1 *= XSimdBatch::load_unaligned(scale + kXSimdBatchSize);
                                           });
        }
        return;
    }

    if (mode == DecodeOutputMode::TopK && top_k <= kFastDecodeTopKMax)
    {
        gemv_topk_packed_impl<kFastDecodeTopKMax>(a,
                                                  packed_b,
                                                  bias,
                                                  n,
                                                  k,
                                                  top_k,
                                                  selection,
                                                  get_decode_gemv_i8_min_parallel_work(),
                                                  [](const int8_t* src) { return load_int8_batch(src); },
                                                  [&](int block, XSimdBatch& sum0, XSimdBatch& sum1) {
                                                      const float* scale = packed_scales + static_cast<size_t>(block) * kKernelNR;
                                                      sum0 *= XSimdBatch::load_unaligned(scale);
                                                      sum1 *= XSimdBatch::load_unaligned(scale + kXSimdBatchSize);
                                                  });
        return;
    }

    gemv_select_packed_impl(a,
                            packed_b,
                            bias,
                            n,
                            k,
                            mode,
                            top_k,
                            selection,
                            get_decode_gemv_i8_min_parallel_work(),
                            [](const int8_t* src) { return load_int8_batch(src); },
                            [&](int block, XSimdBatch& sum0, XSimdBatch& sum1) {
                                const float* scale = packed_scales + static_cast<size_t>(block) * kKernelNR;
                                sum0 *= XSimdBatch::load_unaligned(scale);
                                sum1 *= XSimdBatch::load_unaligned(scale + kXSimdBatchSize);
                            });
}

}  // namespace cpu
}  // namespace minfer
