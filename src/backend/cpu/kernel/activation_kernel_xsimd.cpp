#include "activation_kernel_xsimd.h"
#include "openmp_utils.h"

#include "xsimd/xsimd.hpp"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

void silu_kernel_xsimd(const float* input, float* output, size_t count)
{
    using Batch = xsimd::batch<float>;
    constexpr size_t lanes = Batch::size;

    const Batch one(1.0f);
    const size_t vectorized = (count / lanes) * lanes;
    const long long chunk_count = static_cast<long long>(vectorized / lanes);

#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(static_cast<size_t>(chunk_count), lanes, 1LL << 15, 2))
#endif
    for (long long chunk = 0; chunk < chunk_count; ++chunk)
    {
        const size_t idx = static_cast<size_t>(chunk) * lanes;
        const Batch vx = Batch::load_unaligned(input + idx);
        const Batch vy = vx / (one + xsimd::exp(-vx));
        vy.store_unaligned(output + idx);
    }

    for (size_t idx = vectorized; idx < count; ++idx)
    {
        const float x = input[idx];
        output[idx] = x / (1.0f + std::exp(-x));
    }
}

}  // namespace cpu
}  // namespace minfer
