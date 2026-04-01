#include "rope_kernel.h"
#include "openmp_utils.h"

#include <cmath>
#include <complex>
#include <vector>
#include <xsimd/xsimd.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

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
    std::vector<float> inv_freq(complex_dim);
    std::vector<float> sin_step(complex_dim);
    std::vector<float> cos_step(complex_dim);

    for (int idx = 0; idx < complex_dim; ++idx)
    {
        inv_freq[idx] = 1.0f / std::pow(freq_base, (2.0f * idx) / head_dim);
    }
    build_sincos_cache(1.0f, inv_freq, sin_step, cos_step);

    const size_t q_step = static_cast<size_t>(head_count) * head_dim;
    const size_t k_step = static_cast<size_t>(head_count_kv) * head_dim;

    const bool parallel = should_parallelize_1d_loop(
        static_cast<size_t>(seq_len),
        static_cast<size_t>(head_count + head_count_kv) * static_cast<size_t>(head_dim),
        1LL << 13,
        2);

#ifdef _OPENMP
#pragma omp parallel if(parallel)
#endif
    {
        std::vector<float> sin_cache(complex_dim);
        std::vector<float> cos_cache(complex_dim);
        int prev_seq = -2;

#ifdef _OPENMP
#pragma omp for
#endif
        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx)
        {
            const bool need_rebuild = (seq_idx != prev_seq + 1) || ((seq_idx % kReanchorPeriod) == 0);
            if (need_rebuild)
            {
                build_sincos_cache(static_cast<float>(start_pos + seq_idx), inv_freq, sin_cache, cos_cache);
            }
            else
            {
                advance_sincos_cache(sin_cache, cos_cache, sin_step, cos_step);
            }

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
            prev_seq = seq_idx;
        }
    }
}

}  // namespace cpu
}  // namespace minfer
