#ifndef MINFER_XSIMD_KERNEL_UTILS_H
#define MINFER_XSIMD_KERNEL_UTILS_H

#include "minfer/define.h"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace minfer {
namespace cpu {

using XSimdBatch = xsimd::batch<float>;
constexpr std::size_t kXSimdBatchSize = XSimdBatch::size;

#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
template <std::size_t Lanes = kXSimdBatchSize, typename std::enable_if<Lanes == 4, int>::type = 0>
__attribute__((target("f16c")))
inline XSimdBatch load_hfloat_batch_f16c(const hfloat* src)
{
    const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(static_cast<const void*>(src)));
    return XSimdBatch(_mm_cvtph_ps(raw));
}

template <std::size_t Lanes = kXSimdBatchSize, typename std::enable_if<Lanes == 8, int>::type = 0>
__attribute__((target("avx,f16c")))
inline XSimdBatch load_hfloat_batch_f16c(const hfloat* src)
{
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(static_cast<const void*>(src)));
    return XSimdBatch(_mm256_cvtph_ps(raw));
}
#endif

inline XSimdBatch load_hfloat_batch_scalar(const hfloat* src)
{
    std::array<float, kXSimdBatchSize> tmp{};
    for (std::size_t idx = 0; idx < kXSimdBatchSize; ++idx)
    {
        tmp[idx] = static_cast<float>(src[idx]);
    }
    return XSimdBatch::load_unaligned(tmp.data());
}

inline XSimdBatch load_hfloat_batch(const hfloat* src)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    static_assert(sizeof(hfloat) == sizeof(std::uint16_t), "hfloat must use 16-bit storage");

    if constexpr (kXSimdBatchSize == 8)
    {
        if (__builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c"))
        {
            return load_hfloat_batch_f16c(src);
        }
    }

    if constexpr (kXSimdBatchSize == 4)
    {
        if (__builtin_cpu_supports("f16c"))
        {
            return load_hfloat_batch_f16c(src);
        }
    }
#endif

    return load_hfloat_batch_scalar(src);
}

inline XSimdBatch load_int8_batch(const int8_t* src)
{
#if defined(__GNUC__) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    if constexpr (kXSimdBatchSize == 16)
    {
#if defined(__AVX512F__)
        __m128i v8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
        __m512i v32 = _mm512_cvtepi8_epi32(v8);
        return XSimdBatch(_mm512_cvtepi32_ps(v32));
#endif
    }
    else if constexpr (kXSimdBatchSize == 8)
    {
#if defined(__AVX2__)
        __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
        __m256i v32 = _mm256_cvtepi8_epi32(v8);
        return XSimdBatch(_mm256_cvtepi32_ps(v32));
#endif
    }
    else if constexpr (kXSimdBatchSize == 4)
    {
#if defined(__SSE4_1__)
        int32_t val;
        std::memcpy(&val, src, sizeof(val));
        __m128i v8 = _mm_cvtsi32_si128(val);
        __m128i v32 = _mm_cvtepi8_epi32(v8);
        return XSimdBatch(_mm_cvtepi32_ps(v32));
#endif
    }
#endif
    return xsimd::load_as<float>(src, xsimd::unaligned_mode {});
}

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_XSIMD_KERNEL_UTILS_H
