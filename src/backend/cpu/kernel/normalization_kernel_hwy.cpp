#include "normalization_kernel_hwy.h"

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace minfer {
namespace cpu {

namespace hn = hwy::HWY_NAMESPACE;

void softmax_lastdim_hwy(const float* input, float* output, size_t outer, size_t inner)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));

    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx)
    {
        const float* input_row = input + outer_idx * inner;
        float* output_row = output + outer_idx * inner;

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

void rmsnorm_lastdim_hwy(const float* input,
                         const float* weight,
                         float* output,
                         size_t outer,
                         size_t channels,
                         float eps)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));

    for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx)
    {
        const float* input_row = input + outer_idx * channels;
        float* output_row = output + outer_idx * channels;

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
