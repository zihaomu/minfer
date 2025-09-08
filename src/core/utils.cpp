//
// Created by mzh on 2024/8/5.
//

#include "minfer/utils.h"

namespace minfer {

// based on https://gist.github.com/martin-kallman/5049614
// float32
// Martin Kallman
//
// Fast half-precision to single-precision floating point conversion
//  - Supports signed zero and denormals-as-zero (DAZ)
//  - Does not support infinities or NaN
//  - Few, partially pipelinable, non-branching instructions,
//  - Core opreations ~6 clock cycles on modern x86-64
float fp16_to_fp32(const uint16_t in) {
    uint32_t t1;
    uint32_t t2;
    uint32_t t3;

    t1 = in & 0x7fffu;                       // Non-sign bits
    t2 = in & 0x8000u;                       // Sign bit
    t3 = in & 0x7c00u;                       // Exponent

    t1 <<= 13u;                              // Align mantissa on MSB
    t2 <<= 16u;                              // Shift sign bit into position

    t1 += 0x38000000;                       // Adjust bias

    t1 = (t3 == 0 ? 0 : t1);                // Denormals-as-zero

    t1 |= t2;                               // Re-insert sign bit

    float f = 0.f;
    *((uint32_t *)&f) = t1;
    return f;
//    *((uint32_t *) out) = t1;
}

// float16
// Martin Kallman
//
// Fast single-precision to half-precision floating point conversion
//  - Supports signed zero, denormals-as-zero (DAZ), flush-to-zero (FTZ),
//    clamp-to-max
//  - Does not support infinities or NaN
//  - Few, partially pipelinable, non-branching instructions,
//  - Core opreations ~10 clock cycles on modern x86-64
uint16_t fp32_to_fp16(const float in) {
    uint32_t inu = *((uint32_t * ) & in);
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

    uint16_t u = 0;
    *((uint16_t *)&u) = t1;
    return u;
//    *((uint16_t *) out) = t1;
}

std::string shape_to_str(const Mat& m)
{
    return shape_to_str(m.shape());
}

std::string shape_to_str(const MatShape& shape)
{
    int dims = shape.size();
    const auto& p = shape.data();
    std::string shape_str = "[ " + std::to_string(p[0]);
    for (int i = 1; i < dims; i++)
    {
        shape_str += " x " + std::to_string(p[i]);
    }
    shape_str += "]";

    return shape_str;
}

MatShape get_gemm_shape(const Mat& A, const Mat& B)
{
    return get_gemm_shape(A.shape(), B.shape());
}

MatShape get_gemm_shape(const MatShape& shape_a, const MatShape& shape_b)
{
    // 目前不处理 K x KxN 这种情况。
    M_Assert(shape_a.size() >= 2 && shape_b.size() >= 2 && "Mat shapes on gemm function are miss matching!");

    // generate output shape with brodcast rule
    // need to handle When a and b shape is 1xK x KxN, or MxK x 1xN, or MxK x Kx1
    MatShape shape_c;

    int index_a = shape_a.size() - 2;
    int index_b = shape_b.size() - 2;

    while (index_a > 0 || index_b > 0)
    {
        if (index_a > 0 && index_b > 0)
        {
            if (shape_a[index_a - 1] == shape_b[index_b - 1])
            {
                shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            }
            else if (shape_a[index_a - 1] == 1)
            {
                shape_c.insert(shape_c.begin(), shape_b[index_b - 1]);
            }
            else if (shape_b[index_b - 1] == 1)
            {
                shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            }
            else
            {
                M_Error(NULL, "Mat shapes on gemm function are miss matching!");
            }
            index_a--;
            index_b--;
        }
        else if (index_a > 0)
        {
            shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            index_a--;
        }
        else if (index_b > 0)
        {
            shape_c.insert(shape_c.begin(), shape_b[index_b - 1]);
            index_b--;
        }
        else
            M_Error(NULL, "Mat shapes on gemm function are miss matching!");
    }

    // 如果有一个维度为1维度，说明 出现 MxK x K = M的情况
    if (shape_a.size() == 1 || shape_b.size() == 1)
    {
        if (shape_a.size() == 1 && shape_b.size() == 1)
        {
            shape_c = {1}; // 处理 1xK x Kx1 = 1的情况
        }
    }
    else
    {
        shape_c.push_back(shape_a[shape_a.size() - 2]);
        shape_c.push_back(shape_b[shape_b.size() - 1]);
    }

    return shape_c;
}

}
