#include "rope_kernel.h"
#include "openmp_utils.h"
#include "minfer/parallel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace minfer {
namespace cpu {

namespace {
using ScalarBatch = xsimd::batch<float>;
constexpr size_t kScalarLanes = ScalarBatch::size;
constexpr int kReanchorPeriod = 256;

void build_sincos_cache(float pos,
                        const std::vector<float>& inv_freq,
                        std::vector<float>& sin_cache,
                        std::vector<float>& cos_cache)
{
    const int complex_dim = static_cast<int>(inv_freq.size());
    int pair_idx = 0;
    const ScalarBatch pos_batch(pos);
    for (; pair_idx + static_cast<int>(kScalarLanes) <= complex_dim; pair_idx += static_cast<int>(kScalarLanes))
    {
        const ScalarBatch inv = ScalarBatch::load_unaligned(inv_freq.data() + pair_idx);
        const auto sincos_pair = xsimd::sincos(pos_batch * inv);
        sincos_pair.first.store_unaligned(sin_cache.data() + pair_idx);
        sincos_pair.second.store_unaligned(cos_cache.data() + pair_idx);
    }
    for (; pair_idx < complex_dim; ++pair_idx)
    {
        const float angle = pos * inv_freq[pair_idx];
        sin_cache[pair_idx] = std::sin(angle);
        cos_cache[pair_idx] = std::cos(angle);
    }
}

void advance_sincos_cache(std::vector<float>& sin_cache,
                          std::vector<float>& cos_cache,
                          const std::vector<float>& sin_step,
                          const std::vector<float>& cos_step)
{
    const int complex_dim = static_cast<int>(sin_cache.size());
    int pair_idx = 0;
    for (; pair_idx + static_cast<int>(kScalarLanes) <= complex_dim; pair_idx += static_cast<int>(kScalarLanes))
    {
        const ScalarBatch s = ScalarBatch::load_unaligned(sin_cache.data() + pair_idx);
        const ScalarBatch c = ScalarBatch::load_unaligned(cos_cache.data() + pair_idx);
        const ScalarBatch step_s = ScalarBatch::load_unaligned(sin_step.data() + pair_idx);
        const ScalarBatch step_c = ScalarBatch::load_unaligned(cos_step.data() + pair_idx);

        const ScalarBatch next_s = xsimd::fma(s, step_c, c * step_s);
        const ScalarBatch next_c = c * step_c - s * step_s;

        next_s.store_unaligned(sin_cache.data() + pair_idx);
        next_c.store_unaligned(cos_cache.data() + pair_idx);
    }
    for (; pair_idx < complex_dim; ++pair_idx)
    {
        const float s = sin_cache[pair_idx];
        const float c = cos_cache[pair_idx];
        sin_cache[pair_idx] = s * cos_step[pair_idx] + c * sin_step[pair_idx];
        cos_cache[pair_idx] = c * cos_step[pair_idx] - s * sin_step[pair_idx];
    }
}

struct RopeFreqTable
{
    int head_dim = 0;
    float freq_base = 0.0f;
    std::vector<float> inv_freq;
    std::vector<float> sin_step;
    std::vector<float> cos_step;
};

const RopeFreqTable& get_cached_freq_table(int head_dim, float freq_base)
{
    thread_local RopeFreqTable table;
    const bool cache_hit = (table.head_dim == head_dim) &&
                           (table.freq_base == freq_base) &&
                           (!table.inv_freq.empty());
    if (cache_hit)
    {
        return table;
    }

    const int complex_dim = head_dim / 2;
    table.head_dim = head_dim;
    table.freq_base = freq_base;
    table.inv_freq.resize(complex_dim);
    table.sin_step.resize(complex_dim);
    table.cos_step.resize(complex_dim);

    for (int idx = 0; idx < complex_dim; ++idx)
    {
        table.inv_freq[idx] = 1.0f / std::pow(freq_base, (2.0f * idx) / head_dim);
    }
    build_sincos_cache(1.0f, table.inv_freq, table.sin_step, table.cos_step);
    return table;
}

struct RopeDecodeCache
{
    int head_dim = 0;
    float freq_base = 0.0f;
    int last_pos = -1;
    int steps_since_reanchor = 0;
    std::vector<float> sin_cache;
    std::vector<float> cos_cache;
};

void apply_rope_to_heads(float* data,
                         int head_count,
                         int head_dim,
                         const float* sin_cache,
                         const float* cos_cache)
{
    using Complex = std::complex<float>;
    using ComplexBatch = xsimd::batch<Complex>;
    constexpr size_t lanes = ComplexBatch::size;
    const int complex_dim = head_dim / 2;

    for (int head_idx = 0; head_idx < head_count; ++head_idx)
    {
        Complex* head_complex = reinterpret_cast<Complex*>(data + static_cast<size_t>(head_idx) * head_dim);
        int pair_idx = 0;
        for (; pair_idx + static_cast<int>(lanes) <= complex_dim; pair_idx += static_cast<int>(lanes))
        {
            const ComplexBatch x = ComplexBatch::load_unaligned(head_complex + pair_idx);
            const ComplexBatch rot = ComplexBatch::load_unaligned(cos_cache + pair_idx, sin_cache + pair_idx);
            const ComplexBatch out = x * rot;
            out.store_unaligned(head_complex + pair_idx);
        }
        for (; pair_idx < complex_dim; ++pair_idx)
        {
            head_complex[pair_idx] *= Complex(cos_cache[pair_idx], sin_cache[pair_idx]);
        }
    }
}

}  // namespace

void rope_kernel_inplace(float* q,
                         float* k,
                         int seq_len,
                         int start_pos,
                         int head_count,
                         int head_count_kv,
                         int head_dim,
                         float freq_base)
{
    const int complex_dim = head_dim / 2;
    const RopeFreqTable& freq_table = get_cached_freq_table(head_dim, freq_base);
    const auto& inv_freq = freq_table.inv_freq;
    const auto& sin_step = freq_table.sin_step;
    const auto& cos_step = freq_table.cos_step;

    const size_t q_step = static_cast<size_t>(head_count) * head_dim;
    const size_t k_step = static_cast<size_t>(head_count_kv) * head_dim;

    if (seq_len == 1)
    {
        thread_local RopeDecodeCache decode_cache;
        const bool same_config = (decode_cache.head_dim == head_dim) &&
                                 (decode_cache.freq_base == freq_base);
        if (!same_config || static_cast<int>(decode_cache.sin_cache.size()) != complex_dim)
        {
            decode_cache.head_dim = head_dim;
            decode_cache.freq_base = freq_base;
            decode_cache.last_pos = -1;
            decode_cache.steps_since_reanchor = 0;
            decode_cache.sin_cache.assign(complex_dim, 0.0f);
            decode_cache.cos_cache.assign(complex_dim, 0.0f);
        }

        const bool is_contiguous_step = (decode_cache.last_pos >= 0) && (start_pos == decode_cache.last_pos + 1);
        const bool can_advance = is_contiguous_step && (decode_cache.steps_since_reanchor + 1 < kReanchorPeriod);
        if (can_advance)
        {
            advance_sincos_cache(decode_cache.sin_cache,
                                 decode_cache.cos_cache,
                                 sin_step,
                                 cos_step);
            ++decode_cache.steps_since_reanchor;
        }
        else
        {
            build_sincos_cache(static_cast<float>(start_pos),
                               inv_freq,
                               decode_cache.sin_cache,
                               decode_cache.cos_cache);
            decode_cache.steps_since_reanchor = 0;
        }

        apply_rope_to_heads(q, head_count, head_dim, decode_cache.sin_cache.data(), decode_cache.cos_cache.data());
        apply_rope_to_heads(k, head_count_kv, head_dim, decode_cache.sin_cache.data(), decode_cache.cos_cache.data());
        decode_cache.last_pos = start_pos;
        return;
    }

    const bool parallel = should_parallelize_1d_loop(
        static_cast<size_t>(seq_len),
        static_cast<size_t>(head_count + head_count_kv) * static_cast<size_t>(head_dim),
        1LL << 13,
        2);

    auto process_seq = [&](int seq_idx,
                           std::vector<float>& sin_cache,
                           std::vector<float>& cos_cache) {
        apply_rope_to_heads(q + static_cast<size_t>(seq_idx) * q_step,
                            head_count,
                            head_dim,
                            sin_cache.data(),
                            cos_cache.data());
        apply_rope_to_heads(k + static_cast<size_t>(seq_idx) * k_step,
                            head_count_kv,
                            head_dim,
                            sin_cache.data(),
                            cos_cache.data());
    };

    auto need_rebuild = [&](int seq_idx, int prev_seq) {
        return (seq_idx != prev_seq + 1) || ((seq_idx % kReanchorPeriod) == 0);
    };

    auto process_range = [&](int begin, int end, std::vector<float>& sin_cache, std::vector<float>& cos_cache) {
        int prev_seq = begin - 2;
        for (int seq_idx = begin; seq_idx < end; ++seq_idx)
        {
            if (need_rebuild(seq_idx, prev_seq))
            {
                build_sincos_cache(static_cast<float>(start_pos + seq_idx), inv_freq, sin_cache, cos_cache);
            }
            else
            {
                advance_sincos_cache(sin_cache, cos_cache, sin_step, cos_step);
            }
            process_seq(seq_idx, sin_cache, cos_cache);
            prev_seq = seq_idx;
        }
    };

    if (!parallel)
    {
        std::vector<float> sin_cache(complex_dim);
        std::vector<float> cos_cache(complex_dim);
        process_range(0, seq_len, sin_cache, cos_cache);
        return;
    }

    const int thread_budget = std::max(1, parallel_get_num_threads());
    const long long grain = std::max<long long>(
        1,
        (static_cast<long long>(seq_len) + thread_budget - 1) / thread_budget);

    parallel_for_1d(0, static_cast<long long>(seq_len), grain, [&](long long begin, long long end) {
        std::vector<float> sin_cache(complex_dim);
        std::vector<float> cos_cache(complex_dim);
        process_range(static_cast<int>(begin), static_cast<int>(end), sin_cache, cos_cache);
    });
}

}  // namespace cpu
}  // namespace minfer
