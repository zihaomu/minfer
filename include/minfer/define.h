//
// Created by mzh on 2023/11/2.
//

#ifndef MINFER_DEFINE_H
#define MINFER_DEFINE_H

#include <assert.h>
#include <stdio.h>

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
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

// all Mat type
#define DT_8U   0
#define DT_8S   1
#define DT_16U  2
#define DT_16S  3
#define DT_32S  4
#define DT_32F  5
#define DT_64F  6
#define DT_16F  7
#define DT_16BF 8
#define DT_Bool 9
#define DT_64U  10
#define DT_64S  11
#define DT_32U  12

/** Size of an array/scalar single value, 4 bits per type:
#define DT_8U   - 1 byte
#define DT_8S   - 1 byte
#define DT_16U  - 2
#define DT_16S  - 2
#define DT_32S  - 4
#define DT_32F  - 4
#define DT_64F  - 8
#define DT_16F  - 2
#define DT_16BF - 2
#define DT_Bool - 1
#define DT_64U  - 8
#define DT_64S  - 8
#define DT_32U  - 4
    ...
*/
#define DT_ELEM_SIZE(type) ((int)((0x4881228442211ULL >> (type * 4)) & 15))


// Comparing flag
#define M_CMP_EQ   0
#define M_CMP_GT   1
#define M_CMP_GE   2
#define M_CMP_LT   3
#define M_CMP_LE   4
#define M_CMP_NE   5

#endif //MINFER_DEFINE_H
