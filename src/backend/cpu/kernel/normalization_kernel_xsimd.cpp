#include "normalization_kernel_xsimd.h"
#include "openmp_utils.h"
#include "xsimd_kernel_utils.h"

#include "xsimd/xsimd.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace {

inline float rms_scale_xsimd(const float* input_row, size_t channels, float eps)
{
    XSimdBatch sum_vec(0.0f);
    size_t idx = 0;
    for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
    {
        const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
        sum_vec = xsimd::fma(x, x, sum_vec);
    }

    float sum_sq = xsimd::reduce_add(sum_vec);
    for (; idx < channels; ++idx)
    {
        sum_sq += input_row[idx] * input_row[idx];
    }

    return 1.0f / std::sqrt(sum_sq / static_cast<float>(channels) + eps);
}

inline void online_weighted_accumulate_xsimd(float* out_row,
                                             const float* v_row,
                                             size_t head_dim,
                                             float exp_diff,
                                             float exp_weight)
{
    const XSimdBatch diff_vec(exp_diff);
    const XSimdBatch weight_vec(exp_weight);

    size_t idx = 0;
    for (; idx + kXSimdBatchSize <= head_dim; idx += kXSimdBatchSize)
    {
        const XSimdBatch out_vec = XSimdBatch::load_unaligned(out_row + idx);
        const XSimdBatch v_vec = XSimdBatch::load_unaligned(v_row + idx);
        const XSimdBatch updated = xsimd::fma(v_vec, weight_vec, out_vec * diff_vec);
        updated.store_unaligned(out_row + idx);
    }
    for (; idx < head_dim; ++idx)
    {
        out_row[idx] = out_row[idx] * exp_diff + v_row[idx] * exp_weight;
    }
}

inline void scale_row_xsimd(float* row, size_t len, float inv_scale)
{
    const XSimdBatch inv_vec(inv_scale);

    size_t idx = 0;
    for (; idx + kXSimdBatchSize <= len; idx += kXSimdBatchSize)
    {
        const XSimdBatch vec = XSimdBatch::load_unaligned(row + idx) * inv_vec;
        vec.store_unaligned(row + idx);
    }
    for (; idx < len; ++idx)
    {
        row[idx] *= inv_scale;
    }
}

}  // namespace

void softmax_lastdim_xsimd(const float* input, float* output, size_t outer, size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * inner;
        float* output_row = output + outer_i * inner;

        XSimdBatch max_vec(std::numeric_limits<float>::lowest());
        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= inner; idx += kXSimdBatchSize)
        {
            max_vec = xsimd::max(max_vec, XSimdBatch::load_unaligned(input_row + idx));
        }

        float max_val = xsimd::reduce_max(max_vec);
        for (; idx < inner; ++idx)
        {
            max_val = std::max(max_val, input_row[idx]);
        }

        XSimdBatch sum_vec(0.0f);
        const XSimdBatch max_batch(max_val);
        idx = 0;
        for (; idx + kXSimdBatchSize <= inner; idx += kXSimdBatchSize)
        {
            const XSimdBatch exp_vec = xsimd::exp(XSimdBatch::load_unaligned(input_row + idx) - max_batch);
            exp_vec.store_unaligned(output_row + idx);
            sum_vec += exp_vec;
        }

        float sum_val = xsimd::reduce_add(sum_vec);
        for (; idx < inner; ++idx)
        {
            const float exp_v = std::exp(input_row[idx] - max_val);
            output_row[idx] = exp_v;
            sum_val += exp_v;
        }

        const float inv_sum = 1.0f / sum_val;
        const XSimdBatch inv_sum_vec(inv_sum);
        idx = 0;
        for (; idx + kXSimdBatchSize <= inner; idx += kXSimdBatchSize)
        {
            const XSimdBatch out_vec = XSimdBatch::load_unaligned(output_row + idx) * inv_sum_vec;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < inner; ++idx)
        {
            output_row[idx] *= inv_sum;
        }
    }
}

void causal_masked_softmax_square_xsimd(const float* input,
                                        float* output,
                                        size_t outer,
                                        size_t seq_len,
                                        float scale)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, seq_len * seq_len, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_mat = input + outer_i * seq_len * seq_len;
        float* output_mat = output + outer_i * seq_len * seq_len;
        const XSimdBatch scale_vec(scale);

        for (size_t row = 0; row < seq_len; ++row)
        {
            const size_t valid_cols = row + 1;
            const float* input_row = input_mat + row * seq_len;
            float* output_row = output_mat + row * seq_len;

            XSimdBatch max_vec(std::numeric_limits<float>::lowest());
            size_t idx = 0;
            for (; idx + kXSimdBatchSize <= valid_cols; idx += kXSimdBatchSize)
            {
                const XSimdBatch scaled = XSimdBatch::load_unaligned(input_row + idx) * scale_vec;
                max_vec = xsimd::max(max_vec, scaled);
            }

            float max_val = xsimd::reduce_max(max_vec);
            for (; idx < valid_cols; ++idx)
            {
                max_val = std::max(max_val, input_row[idx] * scale);
            }

            XSimdBatch sum_vec(0.0f);
            const XSimdBatch max_batch(max_val);
            idx = 0;
            for (; idx + kXSimdBatchSize <= valid_cols; idx += kXSimdBatchSize)
            {
                const XSimdBatch scaled = XSimdBatch::load_unaligned(input_row + idx) * scale_vec;
                const XSimdBatch exp_vec = xsimd::exp(scaled - max_batch);
                exp_vec.store_unaligned(output_row + idx);
                sum_vec += exp_vec;
            }

            float sum_val = xsimd::reduce_add(sum_vec);
            for (; idx < valid_cols; ++idx)
            {
                const float exp_v = std::exp(input_row[idx] * scale - max_val);
                output_row[idx] = exp_v;
                sum_val += exp_v;
            }

            const float inv_sum = 1.0f / sum_val;
            const XSimdBatch inv_sum_vec(inv_sum);
            idx = 0;
            for (; idx + kXSimdBatchSize <= valid_cols; idx += kXSimdBatchSize)
            {
                const XSimdBatch out_vec = XSimdBatch::load_unaligned(output_row + idx) * inv_sum_vec;
                out_vec.store_unaligned(output_row + idx);
            }
            for (; idx < valid_cols; ++idx)
            {
                output_row[idx] *= inv_sum;
            }

            std::fill(output_row + valid_cols, output_row + seq_len, 0.0f);
        }
    }
}

void causal_softmax_weighted_sum_square_xsimd(const float* qk,
                                              const float* v,
                                              float* out,
                                              size_t outer,
                                              size_t seq_len,
                                              size_t head_dim,
                                              float scale)
{
    const size_t rows_total = outer * seq_len;
    const long long rows_total_ll = static_cast<long long>(rows_total);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(rows_total, seq_len * head_dim, 1LL << 15, 2))
#endif
    for (long long row_idx_ll = 0; row_idx_ll < rows_total_ll; ++row_idx_ll)
    {
        const size_t row_idx = static_cast<size_t>(row_idx_ll);
        const size_t outer_i = row_idx / seq_len;
        const size_t row = row_idx - outer_i * seq_len;

        const float* qk_mat = qk + outer_i * seq_len * seq_len;
        const float* qk_row = qk_mat + row * seq_len;
        const float* v_mat = v + outer_i * seq_len * head_dim;
        float* out_row = out + row_idx * head_dim;

        std::fill(out_row, out_row + head_dim, 0.0f);

        float m_curr = -std::numeric_limits<float>::infinity();
        float l_curr = 0.0f;

        for (size_t col = 0; col <= row; ++col)
        {
            const float qk_scaled = qk_row[col] * scale;
            const float m_new = std::max(m_curr, qk_scaled);
            const float exp_diff = std::exp(m_curr - m_new);
            const float exp_qk = std::exp(qk_scaled - m_new);

            l_curr = l_curr * exp_diff + exp_qk;

            const float* v_row = v_mat + col * head_dim;
            online_weighted_accumulate_xsimd(out_row, v_row, head_dim, exp_diff, exp_qk);

            m_curr = m_new;
        }

        const float inv_l = 1.0f / l_curr;
        scale_row_xsimd(out_row, head_dim, inv_l);
    }
}

void rmsnorm_lastdim_xsimd_fp16_weight(const float* input,
                                       const hfloat* weight,
                                       float* output,
                                       size_t outer,
                                       size_t channels,
                                       float eps)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        const float scale = rms_scale_xsimd(input_row, channels, eps);
        const XSimdBatch scale_vec(scale);

        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
        {
            const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
            const XSimdBatch w = load_hfloat_batch(weight + idx);
            const XSimdBatch out_vec = x * scale_vec * w;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * static_cast<float>(weight[idx]);
        }
    }
}

void rmsnorm_lastdim_xsimd_i8_weight(const float* input,
                                     const int8_t* weight,
                                     const float* scales,
                                     float* output,
                                     size_t outer,
                                     size_t channels,
                                     float eps)
{
    const float weight_scale = scales[0];
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        const float scale = rms_scale_xsimd(input_row, channels, eps) * weight_scale;
        const XSimdBatch scale_vec(scale);

        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
        {
            const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
            const XSimdBatch w = load_int8_batch(weight + idx);
            const XSimdBatch out_vec = x * scale_vec * w;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * static_cast<float>(weight[idx]);
        }
    }
}

void rmsnorm_lastdim_xsimd(const float* input,
                           const float* weight,
                           float* output,
                           size_t outer,
                           size_t channels,
                           float eps)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        const float scale = rms_scale_xsimd(input_row, channels, eps);
        const XSimdBatch scale_vec(scale);

        size_t idx = 0;
        for (; idx + kXSimdBatchSize <= channels; idx += kXSimdBatchSize)
        {
            const XSimdBatch x = XSimdBatch::load_unaligned(input_row + idx);
            const XSimdBatch w = XSimdBatch::load_unaligned(weight + idx);
            const XSimdBatch out_vec = x * scale_vec * w;
            out_vec.store_unaligned(output_row + idx);
        }
        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * weight[idx];
        }
    }
}

}  // namespace cpu
}  // namespace minfer
