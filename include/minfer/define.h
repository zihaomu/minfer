//
// Created by mzh on 2023/11/2.
//

#ifndef MINFER_DEFINE_H
#define MINFER_DEFINE_H

#include <assert.h>
#include <stdio.h>
#include <cstdint>
#include <memory>

#define M_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#if defined(_MSC_VER)
#if defined(BUILDING_M_DLL)
#define M_PUBLIC __declspec(dllexport)
#elif defined(USING_M_DLL)
#define M_PUBLIC __declspec(dllimport)
#else
#define M_PUBLIC
#endif
#else
#define M_PUBLIC __attribute__((visibility("default")))
#endif
#define STR_IMP(x) #x
#define STR(x) STR_IMP(x)
#define M_VERSION_MAJOR 0
#define M_VERSION_MINOR 0
#define M_VERSION_PATCH 1
#define M_VERSION_STATUS   "-dev"
#define M_VERSION STR(M_VERSION_MAJOR) "." STR(M_VERSION_MINOR) "." STR(M_VERSION_PATCH) M_VERSION_STATUS

#ifdef M_ROOT_PATH
#define M_ROOT STR(M_ROOT_PATH)
#endif
// ERROR CODE
#ifndef M_PI
#define M_PI   3.1415926535897932384626433832795
#endif

#ifndef M_2PI
#define M_2PI  6.283185307179586476925286766559
#endif

#ifndef M_LOG2
#define M_LOG2 0.69314718055994530941723212145818
#endif

/****************************************************************************************\
*                                  Basic Data Type                                       *
\****************************************************************************************/

using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using uint64 = unsigned long int;
using int64 = long int;

/****************************************************************************************\
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

// all Mat type
#define DT_8U   0   // - 1 byte
#define DT_8S   1   // - 1 byte
#define DT_16U  2   // - 2 byte
#define DT_16S  3   // - 2 byte
#define DT_32F  4   // - 4 byte
#define DT_32S  5   // - 4 byte
#define DT_32U  6   // - 4 byte
#define DT_16F  7   // - 2 byte
#define DT_64F  8   // - 8 byte
#define DT_16BF 9   // - 2 byte
#define DT_Bool 10  // - 1 byte
#define DT_64U  11  // - 8 byte
#define DT_64S  12  // - 8 byte

#define DT_MAX  12 // Equal to MAX Data Type

#define DT_ELEM_SIZE(type) ((int)((0x8812824442211ULL >> (type * 4)) & 15))

// Comparing flag
#define M_CMP_EQ   0
#define M_CMP_GT   1
#define M_CMP_GE   2
#define M_CMP_LT   3
#define M_CMP_LE   4
#define M_CMP_NE   5

/****************************************************************************************\
*          exchange-add operation for atomic operations on reference counters            *
\****************************************************************************************/
// take from opencv2/core/cvdef.h Line 706
#ifdef M_XADD
// allow to use user-defined macro
#elif defined __GNUC__ || defined __clang__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)  && !defined __INTEL_COMPILER
#    ifdef __ATOMIC_ACQ_REL
#      define M_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define M_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
// version for gcc >= 4.7
#      define M_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define M_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define M_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
#error "Atomic operations are not supported"
#endif


/****************************************************************************************\
*                                  Float16 Define                                        *
\****************************************************************************************/
#ifdef __cplusplus
class hfloat
{
public:
#if M_WITH_ARM // TODO add arm
    hfloat() : h(0) {}
    explicit hfloat(float x) { h = (__fp16)x; }
    operator float() const { return (float)h; }
    void* get_ptr() { return &h; }
protected:
    __fp16 h;

#else
    void* get_ptr() { return &w; }
    hfloat() : w(0) {}
    explicit hfloat(float x)
    {
        // taken from https://gist.github.com/zhuker/b4bd1fb306c7b04975b712c37c4c4075
        uint32_t inu = *((uint32_t * ) & x);
        uint32_t t1;
        uint32_t t2;
        uint32_t t3;

        t1 = inu & 0x7fffffffu;                 // Non-sign bits
        t2 = inu & 0x80000000u;                 // Sign bit
        t3 = inu & 0x7f800000u;                 // Exponent

        t1 >>= 13u;                             // Align mantissa on MSB
        t2 >>= 16u;                             // Shift sign bit into position

        t1 -= 0x1c000;                         // Adjust bias

        t1 = (t3 < 0x38800000u) ? 0 : t1;       // Flush-to-zero
        t1 = (t3 > 0x8e000000u) ? 0x7bff : t1;  // Clamp-to-max
        t1 = (t3 == 0 ? 0 : t1);               // Denormals-as-zero

        t1 |= t2;                              // Re-insert sign bit

        *((ushort *)&w) = t1;
    }

    operator float() const
    {
#if M_WITH_AVX2 // TODO Add AVX support
        float f;
        _mm_store_ss(&f, _mm_cvtph_ps(_mm_cvtsi32_si128(w)));
        return f;
#else
        float f;
        uint32_t t1;
        uint32_t t2;
        uint32_t t3;

        t1 = w & 0x7fffu;                       // Non-sign bits
        t2 = w & 0x8000u;                       // Sign bit
        t3 = w & 0x7c00u;                       // Exponent

        t1 <<= 13u;                              // Align mantissa on MSB
        t2 <<= 16u;                              // Shift sign bit into position

        t1 += 0x38000000;                       // Adjust bias

        t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

        t1 |= t2;                               // Re-insert sign bit

        *(uint32_t* )(&f) = t1;
        return f;
#endif
    }

protected:
    ushort w;
#endif
};

#else
#error "Fp16 must compile with c++"
#endif



#endif //MINFER_DEFINE_H
