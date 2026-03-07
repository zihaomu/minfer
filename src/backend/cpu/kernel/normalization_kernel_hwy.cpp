#include "normalization_kernel_hwy.h"
#include "openmp_utils.h"

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace hn = hwy::HWY_NAMESPACE;

void softmax_lastdim_hwy(const float* input, float* output, size_t outer, size_t inner)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));

    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * inner;
        float* output_row = output + outer_i * inner;

        auto vmax = hn::Set(d, -std::numeric_limits<float>::infinity());
        size_t idx = 0;
        for (; idx + lanes <= inner; idx += lanes)
        {
            vmax = hn::Max(vmax, hn::LoadU(d, input_row + idx));
        }

        float max_val = hn::ReduceMax(d, vmax);
        for (; idx < inner; ++idx)
        {
            max_val = std::max(max_val, input_row[idx]);
        }

        const auto max_vec = hn::Set(d, max_val);
        auto sum_vec = hn::Zero(d);
        idx = 0;
        for (; idx + lanes <= inner; idx += lanes)
        {
            const auto shifted = hn::Sub(hn::LoadU(d, input_row + idx), max_vec);
            const auto exp_v = hn::Exp(d, shifted);
            hn::StoreU(exp_v, d, output_row + idx);
            sum_vec = hn::Add(sum_vec, exp_v);
        }

        float sum_val = hn::ReduceSum(d, sum_vec);
        for (; idx < inner; ++idx)
        {
            const float exp_v = std::exp(input_row[idx] - max_val);
            output_row[idx] = exp_v;
            sum_val += exp_v;
        }

        const float inv_sum = 1.0f / sum_val;
        const auto inv_sum_vec = hn::Set(d, inv_sum);
        idx = 0;
        for (; idx + lanes <= inner; idx += lanes)
        {
            const auto out_v = hn::LoadU(d, output_row + idx);
            hn::StoreU(hn::Mul(out_v, inv_sum_vec), d, output_row + idx);
        }

        for (; idx < inner; ++idx)
        {
            output_row[idx] *= inv_sum;
        }
    }
}

void causal_masked_softmax_square_hwy(const float* input,
                                      float* output,
                                      size_t outer,
                                      size_t seq_len,
                                      float scale)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));
    const auto scale_vec = hn::Set(d, scale);

    const size_t total_rows = outer * seq_len;
    const long long total_rows_ll = static_cast<long long>(total_rows);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(total_rows, seq_len, 1LL << 14, 2))
#endif
    for (long long row_linear = 0; row_linear < total_rows_ll; ++row_linear)
    {
        const size_t row_linear_i = static_cast<size_t>(row_linear);
        const size_t block_idx = row_linear_i / seq_len;
        const size_t row_idx = row_linear_i % seq_len;
        const size_t valid_cols = row_idx + 1;

        const float* input_row = input + (block_idx * seq_len + row_idx) * seq_len;
        float* output_row = output + (block_idx * seq_len + row_idx) * seq_len;

        auto vmax = hn::Set(d, -std::numeric_limits<float>::infinity());
        size_t idx = 0;
        for (; idx + lanes <= valid_cols; idx += lanes)
        {
            const auto scaled = hn::Mul(hn::LoadU(d, input_row + idx), scale_vec);
            vmax = hn::Max(vmax, scaled);
        }

        float max_val = hn::ReduceMax(d, vmax);
        for (; idx < valid_cols; ++idx)
        {
            max_val = std::max(max_val, input_row[idx] * scale);
        }

        auto sum_vec = hn::Zero(d);
        idx = 0;
        for (; idx + lanes <= valid_cols; idx += lanes)
        {
            const auto scaled = hn::Mul(hn::LoadU(d, input_row + idx), scale_vec);
            const auto exp_v = hn::Exp(d, hn::Sub(scaled, hn::Set(d, max_val)));
            hn::StoreU(exp_v, d, output_row + idx);
            sum_vec = hn::Add(sum_vec, exp_v);
        }

        float sum_val = hn::ReduceSum(d, sum_vec);
        for (; idx < valid_cols; ++idx)
        {
            const float exp_v = std::exp(input_row[idx] * scale - max_val);
            output_row[idx] = exp_v;
            sum_val += exp_v;
        }

        const float inv_sum = 1.0f / sum_val;
        const auto inv_sum_vec = hn::Set(d, inv_sum);
        idx = 0;
        for (; idx + lanes <= valid_cols; idx += lanes)
        {
            const auto out_v = hn::LoadU(d, output_row + idx);
            hn::StoreU(hn::Mul(out_v, inv_sum_vec), d, output_row + idx);
        }
        for (; idx < valid_cols; ++idx)
        {
            output_row[idx] *= inv_sum;
        }
        std::fill(output_row + valid_cols, output_row + seq_len, 0.0f);
    }
}

void rmsnorm_lastdim_hwy_fp16_weight(const float* input,
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

        float sum_sq = 0.0f;
        for (size_t idx = 0; idx < channels; ++idx)
        {
            sum_sq += input_row[idx] * input_row[idx];
        }

        const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(channels) + eps);
        for (size_t idx = 0; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * static_cast<float>(weight[idx]);
        }
    }
}

void rmsnorm_lastdim_hwy_i8_weight(const float* input,
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

        float sum_sq = 0.0f;
        for (size_t idx = 0; idx < channels; ++idx)
        {
            sum_sq += input_row[idx] * input_row[idx];
        }

        const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(channels) + eps);
        for (size_t idx = 0; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * (static_cast<float>(weight[idx]) * weight_scale);
        }
    }
}

void rmsnorm_lastdim_hwy(const float* input,
                         const float* weight,
                         float* output,
                         size_t outer,
                         size_t channels,
                         float eps)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));

    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, channels, 1LL << 14, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* input_row = input + outer_i * channels;
        float* output_row = output + outer_i * channels;

        auto sum_sq_vec = hn::Zero(d);
        size_t idx = 0;
        for (; idx + lanes <= channels; idx += lanes)
        {
            const auto x = hn::LoadU(d, input_row + idx);
            sum_sq_vec = hn::MulAdd(x, x, sum_sq_vec);
        }

        float sum_sq = hn::ReduceSum(d, sum_sq_vec);
        for (; idx < channels; ++idx)
        {
            sum_sq += input_row[idx] * input_row[idx];
        }

        const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(channels) + eps);
        const auto scale_vec = hn::Set(d, scale);

        idx = 0;
        for (; idx + lanes <= channels; idx += lanes)
        {
            const auto x = hn::LoadU(d, input_row + idx);
            const auto w = hn::LoadU(d, weight + idx);
            hn::StoreU(hn::Mul(hn::Mul(x, scale_vec), w), d, output_row + idx);
        }

        for (; idx < channels; ++idx)
        {
            output_row[idx] = input_row[idx] * scale * weight[idx];
        }
    }
}

}  // namespace cpu
}  // namespace minfer
