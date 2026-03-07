#include "activation_kernel_hwy.h"
#include "openmp_utils.h"

#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace hn = hwy::HWY_NAMESPACE;

void silu_kernel_hwy(const float* input, float* output, size_t count)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));
    const auto one = hn::Set(d, 1.0f);
    const auto zero = hn::Zero(d);

    const size_t vectorized = (count / lanes) * lanes;
    const long long chunk_count = static_cast<long long>(vectorized / lanes);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(static_cast<size_t>(chunk_count), lanes, 1LL << 15, 2))
#endif
    for (long long chunk = 0; chunk < chunk_count; ++chunk)
    {
        const size_t idx = static_cast<size_t>(chunk) * lanes;
        const auto vx = hn::LoadU(d, input + idx);
        const auto exp_neg = hn::Exp(d, hn::Sub(zero, vx));
        const auto vy = hn::Div(vx, hn::Add(one, exp_neg));
        hn::StoreU(vy, d, output + idx);
    }

    for (size_t idx = vectorized; idx < count; ++idx)
    {
        const float x = input[idx];
        output[idx] = x / (1.0f + std::exp(-x));
    }
}

}  // namespace cpu
}  // namespace minfer
