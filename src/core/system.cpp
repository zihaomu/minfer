//
// Created by mzh on 2024/8/2.
//

#include "minfer.h"
#include <ostream>
#include <sstream>

namespace minfer
{

const char* mErrorStr(int status)
{
    static char buf[256];

    switch (status) 
    {
        case Error::StsOk:                     return "No Error";
        case Error::StsBackTrace:              return "Backtrace";
        case Error::StsError:                  return "Unspecified error";
        case Error::StsInternal:               return "Internal error";
        case Error::StsNoMem:                  return "Insufficient memory";
        case Error::StsBadArg:                 return "Bad argument";
        case Error::StsNullPtr:                return "Null pointer";
        case Error::StsBadSize:                return "Incorrect size of input array";
        case Error::StsNotImplemented:         return "The function/feature is not implemented";
        case Error::StsAssert:                 return "Assertion failed";
    }

    snprintf(buf, sizeof(buf), "Unknown %s code %d", status >= 0 ? "status":"error", status);
    return buf;
}

int m_vsnprintf(char* buf, int len, const char* fmt, va_list args)
{
#if defined _MSC_VER
    if (len <= 0) return len == 0 ? 1024 : -1;
    int res = _vsnprintf_s(buf, len, _TRUNCATE, fmt, args);
    // ensure null terminating on VS
    if (res >= 0 && res < len)
    {
        buf[res] = 0;
        return res;
    }
    else
    {
        buf[len - 1] = 0; // truncate happened
        return res >= len ? res : (len * 2);
    }
#else
    return vsnprintf(buf, len, fmt, args);
#endif
}

int m_snprintf(char* buf, int len, const char* fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    int res = m_vsnprintf(buf, len, fmt, va);
    va_end(va);
    return res;
}

std::string format( const char* fmt, ... )
{
    std::vector<char> buf(1024);

    for ( ; ; )
    {
        va_list va;
        va_start(va, fmt);
        int bsize = static_cast<int>(buf.size());
        int len = m_vsnprintf(buf.data(), bsize, fmt, va);
        va_end(va);

        M_ASSERT(len >= 0 && "Check format string for errors");
        if (len >= bsize)
        {
            buf.resize(len + 1);
            continue;
        }
        buf[bsize - 1] = 0;
        return std::string(buf.data(), len);
    }
}

Exception::Exception()
{
    code = 0;
    line = 0;
}

Exception::Exception(int _code, const std::string& _err, const std::string _func, const std::string& _file, int _line)
: code(_code), err(_err), func(_func), file(_file), line(_line)
{
    this->formatMessage();
}

Exception::~Exception() noexcept {};

/*!
 \return the error description and the context as a text string.
*/
const char *Exception::what() const noexcept {return msg.c_str();}

void Exception::formatMessage()
{
    size_t pos = err.find('\n');
    bool multiline = pos != std::string::npos;

    if (multiline)
    {
        std::stringstream ss;
        size_t prev_pos = 0;

        while (pos != std::string::npos)
        {
            ss << "> "<<err.substr(prev_pos, pos - prev_pos) << std::endl;
            prev_pos = pos + 1;
            pos = err.find('\n', prev_pos);
        }

        ss << "> "<<err.substr(prev_pos);
        if (err[err.size() - 1] != '\n')
            ss << std::endl;
        err = ss.str();
    }

    if (func.size() > 0)
    {
        if (multiline)
            msg = format("Minfer(%s) %s:%d: error (%d:%s) in function '%s' \n %s",
                         M_VERSION, file.c_str(), line, code, mErrorStr(code), func.c_str(), err.c_str());
        else
            msg = format("Minfer(%s) %s:%d: error (%d:%s) %s in function '%s' \n",
                         M_VERSION, file.c_str(), line, code, mErrorStr(code), func.c_str(), err.c_str());
    }
    else
        msg = format("Minfer(%s) %s:%d: error (%d:%s) %s%s \n",
                     M_VERSION, file.c_str(), line, code, mErrorStr(code), func.c_str(), err.c_str(), multiline ? "" : "\n");

}

static void dumpException(const Exception& exc)
{
    const char* errorStr = mErrorStr(exc.code);
    char buf[1 << 12];

    m_snprintf(buf, sizeof(buf),
                "Minfer(%s) Error: %s (%s) in %s, file %s, line %d",
                M_VERSION,
                errorStr, exc.err.c_str(), exc.func.size() > 0 ?
                                           exc.func.c_str() : "unknown function", exc.file.c_str(), exc.line);
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_ERROR, "minfer::error()", "%s", buf);
#else
    fflush(stdout); fflush(stderr);
    fprintf(stderr, "%s\n", buf);
    fflush(stderr);
#endif
}

void error(const Exception& exc)
{
    throw exc;
#ifdef __GNUC__
# if !defined __clang__ && !defined __APPLE__
    // this suppresses this warning: "noreturn" function does return [enabled by default]
    __builtin_trap();
    // or use infinite loop: for (;;) {}
# endif
#endif
}

void error(int code, const std::string& err, const std::string func, const std::string& file, int line)
{
    error(minfer::Exception(code, err, func, file, line));
#ifdef __GNUC__
# if !defined __clang__ && !defined __APPLE__
    // this suppresses this warning: "noreturn" function does return [enabled by default]
    __builtin_trap();
    // or use infinite loop: for (;;) {}
# endif
#endif
}

}