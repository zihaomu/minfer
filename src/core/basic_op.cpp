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
    binaryFunc(BinaryOp::ADD, a, b, c);
//    preProcessAndCheck(a, b, c);
//
//    int type = a.type();
//
//    size_t totalSize = a.total();
//    if (type == DT_32F)
//    {
//        const float* ap = (const float*)a.data;
//        const float* bp = (const float*)b.data;
//        float* cp = (float*)c.data;
//
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] + bp[i];
//        }
//    }
//    else if (type == DT_32S)
//    {
//        const int* ap = (const int*)a.data;
//        const int* bp = (const int*)b.data;
//        int* cp = (int*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] + bp[i];
//        }
//    }
//    else
//        M_ERROR(NULL, "Unsupported format at function \" add \" type = %d!", type);
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

    binaryFunc(BinaryOp::SUB, a, b, c);
//
//    preProcessAndCheck(a, b, c);
//
//    int type = a.type();
//
//    size_t totalSize = a.total();
//    if (type == DT_32F)
//    {
//        const float* ap = (const float*)a.data;
//        const float* bp = (const float*)b.data;
//        float* cp = (float*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] - bp[i];
//        }
//    }
//    else if (type == DT_32S)
//    {
//        const int* ap = (const int*)a.data;
//        const int* bp = (const int*)b.data;
//        int* cp = (int*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] - bp[i];
//        }
//    }
//    else
//        M_ERROR(NULL, "Unsupported format at function \" subtract \" type = %d!", type);
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
    binaryFunc(BinaryOp::MUL, a, b, c);
//
//    preProcessAndCheck(a, b, c);
//
//    int type = a.type();
//
//    size_t totalSize = a.total();
//    if (type == DT_32F)
//    {
//        const float* ap = (const float*)a.data;
//        const float* bp = (const float*)b.data;
//        float* cp = (float*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] * bp[i];
//        }
//    }
//    else if (type == DT_32S)
//    {
//        const int* ap = (const int*)a.data;
//        const int* bp = (const int*)b.data;
//        int* cp = (int*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] * bp[i];
//        }
//    }
//    else
//        M_ERROR(NULL, "Unsupported format at function \" multiply \" type = %d!", type);
}

void divide(const Mat& a, const Mat& b, Mat& c)
{
    binaryFunc(BinaryOp::DIV, a, b, c);

//    preProcessAndCheck(a, b, c);
//
//    int type = a.type();
//
//    size_t totalSize = a.total();
//    if (type == DT_32F)
//    {
//        const float* ap = (const float*)a.data;
//        const float* bp = (const float*)b.data;
//        float* cp = (float*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] / bp[i];
//        }
//    }
//    else if (type == DT_32S)
//    {
//        const int* ap = (const int*)a.data;
//        const int* bp = (const int*)b.data;
//        int* cp = (int*)c.data;
//        for (size_t i = 0; i < totalSize; i++)
//        {
//            cp[i] = ap[i] / bp[i];
//        }
//    }
//    else
//        M_ERROR(NULL, "Unsupported format at function \" divide \" type = %d!", type);
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

// Helper contains all information used in binary BinaryOp.
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
            else if (shape0[idx_0] == 1 || idx_0 < 0)
            {
                out_shape[idx] = shape1[idx_1];
                inp1_shape_align[idx] = out_shape[idx];
            }
            else if (shape1[idx_1] == 1 || idx_1 < 0)
            {
                out_shape[idx] = shape0[idx_0];
                inp0_shape_align[idx] = out_shape[idx];
            }
            else
            {
                std::string str_0 = shape_to_str(a);
                std::string str_1 = shape_to_str(b);
                std::string log_info = "Broadcasting error! The two input shape are" + str_0 + " and " + str_1;
                M_Error(Error::StsBadSize, log_info.c_str());
            }

            idx_0--;
            idx_1--;
            idx--;
        }

        // set dim steps
        auto get_step_func = [](const MatShape& i_s, MatShape& o_s) {
            o_s.resize(i_s.size(), 1);

            // step skip the block_size dimension.
            for (int i = o_s.size() - 3; i >= 0; i--)
            {
                o_s[i] *= i_s[i+1] * o_s[i+1];
            }
        };

        get_step_func(inp0_shape_align, inp0_steps);
        get_step_func(inp1_shape_align, inp1_steps);
        get_step_func(out_shape, out_steps);

        block_num = total(out_shape, 0, max_dims - 1);
        block_size = out_shape[max_dims - 1];
        isInit = true; // set isInit as true.
    }
};



template<typename T, typename Func>
void binary_forward(const Func& op, const BinaryOpHelper& helper,  const uchar* inp0, const uchar* inp1, uchar* out)
{
    M_Assert(helper.isInit && "BinaryOp has not been inited!");

    int max_dims = helper.max_dims;
    int block_size = helper.out_shape[helper.max_dims - 1];
    size_t total_num = total(helper.out_shape);
    int block_num = total_num / block_size;

    M_Assert(total_num % block_size == 0);
    const int esz = sizeof(T); // element size

    const int inner_0 = helper.inp0_shape_align[max_dims - 1] == 1 ? 0 : 1;
    const int inner_1 = helper.inp1_shape_align[max_dims - 1] == 1 ? 0 : 1;

    for (int bi = 0; bi < block_num; bi++)
    {
        // step 0: get output pointer
        T* p_o = (T*)(out + bi * block_size * esz);
        size_t jump0 = 0;
        size_t jump1 = 0;

        int idx = bi;
        for (int k = max_dims - 2; k >= 0; k--)
        {
            int next_idx = idx / helper.out_shape[k];
            int ik = idx - next_idx * helper.out_shape[k];
            jump0 += ik * helper.inp0_steps[k];
            jump1 += ik * helper.inp1_steps[k];
            idx = next_idx;
        }

        T* p_i0 = (T* )(inp0 + jump0 * esz);
        T* p_i1 = (T* )(inp1 + jump1 * esz);

        for (int i = 0; i < block_size; i++, p_o++, p_i0 += inner_0, p_i1 += inner_1)
        {
            *p_o = op(*p_i0, *p_i1);
        }
    }
}

template<typename T, typename... Args>
inline void opDispatch(const BinaryOp op, Args&&... args)
{
    if (std::is_same<T, float>::value)
    {
        M_Assert(op != BinaryOp::MOD && op != BinaryOp::AND && op != BinaryOp::OR && op != BinaryOp::XOR);
    }

    switch (op)
    {
        case BinaryOp::EQUAL:
        {
            auto equal = [](const T &a, const T &b) { return a == b; };
            binary_forward<T>(equal, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::GREATER:
        {
            auto greater = [](const T &a, const T &b) { return a > b; };
            binary_forward<T>(greater, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::GREATER_EQUAL:
        {
            auto greater_equal = [](const T &a, const T &b) { return a >= b; };
            binary_forward<T>(greater_equal, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::LESS:
        {
            auto less = [](const T &a, const T &b) { return a < b; };
            binary_forward<T>(less, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::LESS_EQUAL:
        {
            auto less_equal = [](const T &a, const T &b) { return a <= b; };
            binary_forward<T>(less_equal, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::POW:
        {
            auto pow = [] (const T& a, const T& b) { return std::pow(a, b); };
            binary_forward<T>(pow, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::BITSHIFT:
        {
            auto bitshift = [] (const uint8_t &a, const uint8_t &b) { return a << b; };
            binary_forward<T>(bitshift, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::MOD:
        {
            auto mod = [](const uint8_t &a, const uint8_t &b) { return a % b; };
            binary_forward<T>(mod, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::MUL:
        {
            auto mul = [](const T &a, const T &b) { return a * b; };
            binary_forward<T>(mul, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::SUB:
        {
            auto sub = [](const T &a, const T &b) { return a - b; };
            binary_forward<T>(sub, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::ADD:
        {
            auto add = [](const T &a, const T &b) { return a + b; };
            binary_forward<T>(add, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::DIV:
        {
            auto div = [](const T &a, const T &b) { return a / b; };
            binary_forward<T>(div, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::AND:
        {
            auto op_and = [](const uint8_t &a, const uint8_t &b) { return a & b; };
            binary_forward<T>(op_and, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::OR:
        {
            auto op_or = [](const uint8_t &a, const uint8_t &b) { return a | b; };
            binary_forward<T>(op_or, std::forward<Args>(args)...);
            break;
        }
        case BinaryOp::XOR:
        {
            auto op_xor = [](const uint8_t &a, const uint8_t &b) { return a ^ b; };
            binary_forward<T>(op_xor, std::forward<Args>(args)...);
            break;
        }
        default:
            M_Error_(Error::StsBadType, ("Unsupported op on Mat binary function! Op = %d!", (int)op));
    }
}

template<typename... Args>
inline void typeDispatch(const int type, Args&&... args)
{
    switch (type)
    {
        case DT_8U:
            opDispatch<uint8_t>(std::forward<Args>(args)...);
            break;
        case DT_32S:
            opDispatch<int32_t>(std::forward<Args>(args)...);
            break;
        case DT_32F:
            opDispatch<float>(std::forward<Args>(args)...);
            break;
        default:
            M_Error_(Error::StsBadType, ("Unsupported type on Mat binary function! Type = %d!", type));
    }
}

void binaryFunc(BinaryOp op, const Mat& a, const Mat& b, Mat& c)
{
    M_Assert(a.type() == b.type());
    BinaryOpHelper helper = BinaryOpHelper();
    helper.init(a, b);

    c = Mat(helper.out_shape, a.type());

    typeDispatch(a.type(), op, helper, a.data, b.data, c.data);
}

}
