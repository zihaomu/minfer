#include "minfer.h"

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

void broad_cast_(const Mat& a, const Mat& b, Mat& c)
{

}

/****************************************************************************************\
*                                  Binary Op Implementation                              *
\****************************************************************************************/

/* Design Note: should Binary Op always return the same type?
 *
 * */
// Helper contains all information used in binary operation.
class BinaryOpHelper
{
public:
    int max_dims;
    MatShape inp0_shape_align;   // contains new shape of input 0
    MatShape inp1_shape_align;
    MatShape out_shape;

    MatShape inp0_steps; // steps store all dimension jumping numbers of pointer.
    MatShape inp1_steps;
    MatShape out_steps;

    /* Reorganize mat for [block_num x block_size] based on out_shape.
     * For example: out shape is [a x b x c], the [a x b] is block_num, and c is inner block_size
     * */
    size_t block_size;
    size_t block_num;

    bool isInit = false;

    void init(const Mat& a, const Mat& b)
    {
        M_Assert(a.type() == b.type() && "Input data type is different!");

        MatShape shape0 = a.shape();
        MatShape shape1 = b.shape();

        int dim_0 = shape0.size();
        int dim_1 = shape1.size();

        max_dims = std::max(shape0.size(), shape1.size());

        // broadcasting the shape
        inp0_shape_align.resize(max_dims, 1);
        inp1_shape_align.resize(max_dims, 1);
        out_shape.resize(max_dims, 1);

        int idx_0 = dim_0 - 1;
        int idx_1 = dim_1 - 1;
        int idx = max_dims - 1; // shape loop cur

        while (idx >= 0)
        {
            if (shape0[idx_0] == shape1[idx_1])
            {
                out_shape[idx] = shape0[idx_0];
                inp0_shape_align[idx] = shape0[idx_0];
                inp1_shape_align[idx] = shape0[idx_0];
            }
            else if (shape0[idx_0] == 1 || shape0[idx_0] < 0)
            {
                out_shape[idx] = shape1[idx_1];
                inp1_shape_align[idx] = out_shape[idx];
            }
            else if (shape1[idx_1] == 1 || shape1[idx_1] < 0)
            {
                out_shape[idx] = shape0[idx_0];
                inp0_shape_align[idx] = out_shape[idx];
            }
            else
            {
                std::string str_0 = shape_to_str(a);
                std::string str_1 = shape_to_str(b);
                M_Error_(Error::StsBadSize, ("Broadcasting error! The two input shape are %s and %s !", str_0.c_str(), str_1.c_str()));
            }

            idx_0--;
            idx_1--;
            idx--;
        }

        // set dim steps
        auto get_step_func = [](const MatShape& i_s, MatShape& o_s) {
            o_s.resize(i_s.size(), 1);

            // TODO check 3 or 2
            for (int i = o_s.size() - 3; i >= 0; i--)
            {
                o_s[i] *= i_s[i+1] * o_s[i+1];
            }
        };

        get_step_func(inp0_shape_align, inp0_steps);
        get_step_func(inp1_shape_align, inp1_steps);
        get_step_func(out_shape, out_steps);

        block_num = total(out_shape, 0, max_dims - 1);
        block_size = out_steps[max_dims - 1];
        isInit = true; // set isInit as true.
    }
};

class BinaryOp
{
    BinaryOpHelper helper;
public:
    enum class Operator
    {
        AND = 0,
        EQUAL,
        GREATER,
        GREATER_EQUAL,
        LESS,
        LESS_EQUAL,
        OR,
        POW,
        XOR,
        BITSHIFT,
        MAX,
        MEAN,
        MIN,
        MOD,  // Integer Mod. Reminder's sign = Divisor's sign.
        FMOD, // Floating-point Mod. Reminder's sign = Dividend's sign.
        PROD,
        SUB,
        SUM,
        ADD,
        DIV,
    } op;

    BinaryOp()
    {

    }

    void init();

    template<typename T, typename Func>
    void forward_impl(const Func& op, const char* inp0, const char* inp1, char* out)
    {
        M_Assert(helper.isInit && "BinaryOp has not been inited!");


        int max_dims = helper.max_dims;
        int block_size = helper.out_shape[helper.max_dims - 1];
        size_t total_num = total(helper.out_shape);
        int block_num = total_num / block_size;

        M_Assert(total_num % block_size == 0);

         for (int bi = 0; bi < block_num; bi++)
         {
             // step 0: get output pointer
             T* p_o = (T*)(out + bi * block_size);
             T* p_i0 = (T* )inp0;
             T* p_i1 = (T* )inp1;

             // TODO check 3 or 2
             for (int k = max_dims - 3; k >= 0; k--)
             {

             }
         }
    }

    void forward(const Mat& a, const Mat& b)
    {
        helper = BinaryOpHelper();
        helper.init(a, b);
    }
};

}
