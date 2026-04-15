#include "activation_kernel_xsimd.h"
#include "openmp_utils.h"
#include "minfer/parallel.h"

#include "xsimd/xsimd.hpp"

#include <cmath>

namespace minfer {
namespace cpu {

void silu_kernel_xsimd(const float* input, float* output, size_t count)
{
    using Batch = xsimd::batch<float>;
    constexpr size_t lanes = Batch::size;

    const Batch one(1.0f);
    const size_t vectorized = (count / lanes) * lanes;
    const long long chunk_count = static_cast<long long>(vectorized / lanes);
    const bool parallel =
        should_parallelize_1d_loop(static_cast<size_t>(chunk_count), lanes, 1LL << 15, 2);

    auto process_chunks = [&](long long begin, long long end) {
        for (long long chunk = begin; chunk < end; ++chunk)
        {
            const size_t idx = static_cast<size_t>(chunk) * lanes;
            const Batch vx = Batch::load_unaligned(input + idx);
            const Batch vy = vx / (one + xsimd::exp(-vx));
            vy.store_unaligned(output + idx);
        }
    };

    if (parallel)
    {
        parallel_for_1d(0, chunk_count, 1, process_chunks);
    }
    else
    {
        process_chunks(0, chunk_count);
    }

    for (size_t idx = vectorized; idx < count; ++idx)
    {
        const float x = input[idx];
        output[idx] = x / (1.0f + std::exp(-x));
    }
}

}  // namespace cpu
}  // namespace minfer
