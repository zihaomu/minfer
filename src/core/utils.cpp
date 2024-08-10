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

std::string& shape_to_str(const Mat& m)
{
    return shape_to_str(m.shape());
}

std::string& shape_to_str(const MatShape& shape)
{
    int dims = shape.size();
    const auto& p = shape.data();
    std::string shape_str = "[ " + std::to_string(p[0]) + " x ";
    std::cout<<"shape = ["<<p[0]<<"x";
    for (int i = 1; i < dims; i++)
    {
        shape_str += std::to_string(p[i]) + " x ";
    }
    std::cout<<"]"<<std::endl;

    return shape_str;
}

}
