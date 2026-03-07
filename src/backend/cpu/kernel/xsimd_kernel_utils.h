#ifndef MINFER_XSIMD_KERNEL_UTILS_H
#define MINFER_XSIMD_KERNEL_UTILS_H

#include "minfer/define.h"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace minfer {
namespace cpu {

using XSimdBatch = xsimd::batch<float>;
constexpr std::size_t kXSimdBatchSize = XSimdBatch::size;

inline XSimdBatch load_hfloat_batch(const hfloat* src)
{
    std::array<float, kXSimdBatchSize> tmp{};
    for (std::size_t idx = 0; idx < kXSimdBatchSize; ++idx)
    {
        tmp[idx] = static_cast<float>(src[idx]);
    }
    return XSimdBatch::load_unaligned(tmp.data());
}

inline XSimdBatch load_int8_batch(const int8_t* src)
{
    return xsimd::load_as<float>(src, xsimd::unaligned_mode {});
}

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_XSIMD_KERNEL_UTILS_H
