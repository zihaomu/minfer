//
// This file is made for log and exception system.
// Created by mzh on 2024/8/2.
//

#ifndef MINFER_SYSTEM_H
#define MINFER_SYSTEM_H

#ifndef __cplusplus
#  error system.hpp header must be compiled as C++
#endif

#include <cstdio>

#include "define.h"
#include <exception>
#include <cstdlib>
#include <string>

#if defined(__GNUC__) || defined(__clang__) // at least GCC 3.1+, clang 3.5+
#  if defined(__MINGW_PRINTF_FORMAT)  // https://sourceforge.net/p/mingw-w64/wiki2/gnu%20printf/.
#    define M_FORMAT_PRINTF(string_idx, first_to_check) __attribute__ ((format (__MINGW_PRINTF_FORMAT, string_idx, first_to_check)))
#  else
#    define M_FORMAT_PRINTF(string_idx, first_to_check) __attribute__ ((format (printf, string_idx, first_to_check)))
#  endif
#else
#  define M_FORMAT_PRINTF(A, B)
#endif


#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#define M_BUILD_FOR_IOS
#endif
#endif

// Log system
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

#ifdef M_DEBUG
#define M_PRINT_DBG(format, ...) printf(format, ##__VA_ARGS__)
#else
#define M_PRINT_DBG(format, ...)
#endif

#ifdef M_Func
// keep current value (through OpenCV port file)
#elif defined __GNUC__ || (defined (__cpluscplus) && (__cpluscplus >= 201103))
#define M_Func __func__
#elif defined __clang__ && (__clang_minor__ * 100 + __clang_major__ >= 305)
#define M_Func __func__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION >= 199901)
#define M_Func __func__
#elif defined _MSC_VER
#define M_Func __FUNCTION__
#elif defined(__INTEL_COMPILER) && (_INTEL_COMPILER >= 600)
#define M_Func __FUNCTION__
#elif defined __IBMCPP__ && __IBMCPP__ >=500
#define M_Func __FUNCTION__
#elif defined __BORLAND__ && (__BORLANDC__ >= 0x550)
#define M_Func __FUNC__
#else
#define M_Func "<unknown>"
#endif


namespace minfer
{

namespace Error {
    enum Code {
        StsOk=                       0,  //!< everything is ok
        StsBackTrace=               -1,  //!< pseudo error for back trace
        StsError=                   -2,  //!< unknown /unspecified error
        StsInternal=                -3,  //!< internal error (bad state)
        StsNoMem=                   -4,  //!< insufficient memory
        StsBadArg=                  -5,  //!< function arg/param is bad
        StsBadFunc=                 -6,  //!< unsupported function
        StsNullPtr=                 -7,  //!< null pointer
        StsBadSize=                 -8, //!< the input/output structure size is incorrect
        StsNotImplemented=          -9, //!< the requested function/feature is not implemented
        StsAssert=                 -10, //!< assertion failed
    };
}

class Exception : public std::exception
{
public:
    // Default constructor
    Exception();

    Exception(int _code, const std::string& _err, const std::string _func, const std::string& _file, int _line);

    virtual ~Exception() noexcept;
    virtual const char* what() const noexcept override;
    void formatMessage();

    std::string msg; ///< the formatted error message

    int code;
    std::string err;
    std::string func;
    std::string file;
    int line;
};

void error(const Exception& exc);

void error(int _code, const std::string& _err, const std::string _func, const std::string& _file, int _line);

void mprintf(const char* fmt, ...);

/** @brief Returns a text string formatted using the printf-like expression.
 * This function acts like sprintf but forms and returns and STL string. It can be used to form an error
 * message in Exception constructor.
 *
 * @param fmt printf-compatible formatting specifiers
 * @param ...
 * @return
 *
 **Note**:
|Type|Specifier|
|-|-|
|`const char*`|`%s`|
|`char`|`%c`|
|`float` / `double`|`%f`,`%g`|
|`int`, `long`, `long long`|`%d`, `%ld`, ``%lld`|
|`unsigned`, `unsigned long`, `unsigned long long`|`%u`, `%lu`, `%llu`|
|`uint64` -> `uintmax_t`, `int64` -> `intmax_t`|`%ju`, `%jd`|
|`size_t`|`%zu`|
 */
std::string format(const char* fmt, ...) M_FORMAT_PRINTF(1, 2);


/* This function is made for report error.
 * */
#define M_Error(code, msg) minfer::error(code, msg, M_Func, __FILE__, __LINE__)

/**  @brief Call the error handler.

This macro can be used to construct an error message on-fly to include some dynamic information,
for example:
@code
    // note the extra parentheses around the formatted text message
    CV_Error_(Error::StsOutOfRange,
    ("the value at (%d, %d)=%g is out of range", badPt.x, badPt.y, badValue));
@endcode
@param code one of Error::Code
@param args printf-like formatted error message in parentheses
*/
#define M_Error_( code, args ) minfer::error( code, minfer::format args, M_Func, __FILE__, __LINE__ )

/** @brief Checks a condition at runtime and throws exception if it fails

The macros M_Assert evaluate the specified expression. If it is 0, the macros
raise an error (see minfer::error). The macro M_Assert checks the condition in both Debug and Release.
*/
#define M_Assert( expr ) do { if(!!(expr)) ; else minfer::error( minfer::Error::StsAssert, #expr, M_Func, __FILE__, __LINE__ ); } while(0)
}

#endif //MINFER_SYSTEM_H
