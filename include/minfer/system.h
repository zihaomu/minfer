//
// This file is made for log and exception system.
// Created by mzh on 2024/8/2.
//

#ifndef MINFER_SYSTEM_H
#define MINFER_SYSTEM_H

#include "define.h"
#include <exception>
#include <cstdlib>
#include <string>

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
#define M_Error(code, msg) minfer::error(code, msg, CV_Func, __FILE__, __LINE__)
}

#endif //MINFER_SYSTEM_H
