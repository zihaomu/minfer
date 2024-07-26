//
// Created by mzh on 2023/11/2.
//

#ifndef MINFER_DEFINE_H
#define MINFER_DEFINE_H

#include <assert.h>
#include <stdio.h>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define M_BUILD_FOR_IOS
#endif
#endif

#ifdef M_USE_LOGCAT
#include <android/log.h>
#define M_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MJNI", format, ##__VA_ARGS__)
#define M_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MJNI", format, ##__VA_ARGS__)
#elif defined M_BUILD_FOR_IOS
// on iOS, stderr prints to XCode debug area and syslog prints Console. You need both.
#include <syslog.h>
#define M_PRINT(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#define M_ERROR(format, ...) syslog(LOG_WARNING, format, ##__VA_ARGS__); fprintf(stderr, format, ##__VA_ARGS__)
#else
#define M_PRINT(format, ...) printf(format, ##__VA_ARGS__)
#define M_ERROR(format, ...) printf(format, ##__VA_ARGS__)
#endif

#define M_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define M_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            M_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }


#ifdef DEBUG
#define M_DbgASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            M_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define M_DbgASSERT(x)
#endif

#define FUNC_PRINT(x) MU_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define FUNC_PRINT_ALL(x, type) MU_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define M_CHECK(success, log) \
if(!(success)){ \
MU_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}

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
#define M_VERSION STR(MU_VERSION_MAJOR) "." STR(MU_VERSION_MINOR) "." STR(MU_VERSION_PATCH)

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
