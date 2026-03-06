#include "rope_kernel.h"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace {

void apply_rope_to_heads(float* data,
                         int head_count,
                         int head_dim,
                         const std::vector<float>& sin_cache,
                         const std::vector<float>& cos_cache)
{
    for (int head_idx = 0; head_idx < head_count; ++head_idx)
    {
        float* head_ptr = data + static_cast<size_t>(head_idx) * head_dim;
        for (int pair_idx = 0; pair_idx < head_dim / 2; ++pair_idx)
        {
            const float real = head_ptr[pair_idx * 2];
            const float imag = head_ptr[pair_idx * 2 + 1];
            const float sin_v = sin_cache[pair_idx];
            const float cos_v = cos_cache[pair_idx];

            head_ptr[pair_idx * 2] = real * cos_v - imag * sin_v;
            head_ptr[pair_idx * 2 + 1] = real * sin_v + imag * cos_v;
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

    for (int idx = 0; idx < complex_dim; ++idx)
    {
        inv_freq[idx] = 1.0f / std::pow(freq_base, (2.0f * idx) / head_dim);
    }

    const size_t q_step = static_cast<size_t>(head_count) * head_dim;
    const size_t k_step = static_cast<size_t>(head_count_kv) * head_dim;

#ifdef _OPENMP
#pragma omp parallel if(static_cast<long long>(seq_len) * (head_count + head_count_kv) * head_dim >= (1LL << 13) && !omp_in_parallel())
    {
        std::vector<float> sin_cache(complex_dim);
        std::vector<float> cos_cache(complex_dim);
#pragma omp for
#endif
    for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx)
    {
        const float pos = static_cast<float>(start_pos + seq_idx);
        for (int pair_idx = 0; pair_idx < complex_dim; ++pair_idx)
        {
            const float angle = pos * inv_freq[pair_idx];
            sin_cache[pair_idx] = std::sin(angle);
            cos_cache[pair_idx] = std::cos(angle);
        }

        apply_rope_to_heads(q + static_cast<size_t>(seq_idx) * q_step,
                            head_count,
                            head_dim,
                            sin_cache,
                            cos_cache);
        apply_rope_to_heads(k + static_cast<size_t>(seq_idx) * k_step,
                            head_count_kv,
                            head_dim,
                            sin_cache,
                            cos_cache);
    }
#ifdef _OPENMP
    }
#endif
}

}  // namespace cpu
}  // namespace minfer
