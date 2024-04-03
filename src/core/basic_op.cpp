#include "minfer/basic_op.h"

namespace minfer
{

inline void checkIsSameType(const Mat& a, const Mat& b)
{
    if (!(a.type() == b.type() && a.size == b.size))
        assert(0 && "The input Mats have different type or size!");
}

inline void checkIsSameType(const Mat& a, const Mat& b, const Mat& c)
{
    if (!((a.type() == b.type() && a.size == b.size) && (a.type() == c.type() && a.size == c.size)))
        assert(0 && "The input Mats have different type or size!");
}

inline
void preProcessAndCheck(const Mat& a, const Mat& b, Mat& c)
{
    if (c.empty())
    {
        checkIsSameType(a, b);

        c = Mat(a.dims, a.size.p, a.type());
    }
    else
        checkIsSameType(a, b, c);
}

inline
void preProcessAndCheck(const Mat& a, Mat& c)
{
    if (c.empty())
    {
        assert(!a.empty() && "The input mat can not be empty!");
        c = Mat(a.dims, a.size.p, a.type());
    }
    else
        checkIsSameType(a, c);
}

void add(const Mat& a, const Mat& b, Mat& c)
{
    preProcessAndCheck(a, b, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == DT_32F)
    {
        const float* ap = (const float*)a.data;
        const float* bp = (const float*)b.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] + bp[i];
        }
    }
    else if (type == DT_32S)
    {
        const int* ap = (const int*)a.data;
        const int* bp = (const int*)b.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] + bp[i];
        }
    }
    else
        M_ERROR(NULL, "Unsupported format at function \" add \" type = %d!", type);
}

void addWeighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& c)
{
    preProcessAndCheck(a, b, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == DT_32F)
    {
        const float* ap = (const float*)a.data;
        const float* bp = (const float*)b.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = (float )(ap[i] * alpha + bp[i] * beta);
        }
    }
    else if (type == DT_32S)
    {
        const int* ap = (const int*)a.data;
        const int* bp = (const int*)b.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = (int )(ap[i] * alpha + bp[i] * beta);
        }
    }
    else
        M_ERROR(NULL, "Unsupported format at function \" addWeighted \" type = %d!", type);
}

void subtract(const Mat& a, const Mat& b, Mat& c)
{
    if (a.empty() && !b.empty())
    {
        subtract(b, c);
        return;
    }

    preProcessAndCheck(a, b, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == DT_32F)
    {
        const float* ap = (const float*)a.data;
        const float* bp = (const float*)b.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] - bp[i];
        }
    }
    else if (type == DT_32S)
    {
        const int* ap = (const int*)a.data;
        const int* bp = (const int*)b.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] - bp[i];
        }
    }
    else
        M_ERROR(NULL, "Unsupported format at function \" subtract \" type = %d!", type);
}

void subtract(const Mat& a, Mat& c)
{
    preProcessAndCheck(a, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == DT_32F)
    {
        const float* ap = (const float*)a.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = -ap[i];
        }
    }
    else if (type == DT_32S)
    {
        const int* ap = (const int*)a.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = -ap[i];
        }
    }
    else
        M_ERROR(NULL, "Unsupported format at function \" subtract \" type = %d!", type);
}

void multiply(const Mat& a, const Mat& b, Mat& c)
{
    preProcessAndCheck(a, b, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == DT_32F)
    {
        const float* ap = (const float*)a.data;
        const float* bp = (const float*)b.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] * bp[i];
        }
    }
    else if (type == DT_32S)
    {
        const int* ap = (const int*)a.data;
        const int* bp = (const int*)b.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] * bp[i];
        }
    }
    else
        M_ERROR(NULL, "Unsupported format at function \" multiply \" type = %d!", type);
}

void divide(const Mat& a, const Mat& b, Mat& c)
{
    preProcessAndCheck(a, b, c);

    int type = a.type();

    size_t totalSize = a.total();
    if (type == DT_32F)
    {
        const float* ap = (const float*)a.data;
        const float* bp = (const float*)b.data;
        float* cp = (float*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] / bp[i];
        }
    }
    else if (type == DT_32S)
    {
        const int* ap = (const int*)a.data;
        const int* bp = (const int*)b.data;
        int* cp = (int*)c.data;
        for (size_t i = 0; i < totalSize; i++)
        {
            cp[i] = ap[i] / bp[i];
        }
    }
    else
        M_ERROR(NULL, "Unsupported format at function \" divide \" type = %d!", type);
}

void compare(const Mat& a, const Mat& b, Mat& c, int op)
{
    // TODO
    M_ERROR(NULL, "Un-implemented function at compare!");
}

}
