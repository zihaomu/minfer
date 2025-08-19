//
// Created by mzh on 2024/5/23.
//

#ifndef MINFER_GGML_QUANT_H
#define MINFER_GGML_QUANT_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include "minfer/net.h"
#include "gguf_loader.h"

namespace minfer {

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif // GGML_QKK_64

typedef uint16_t ggml_fp16_t;

// Currently, we only support partial llama.cpp quantized data type.

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Define different quantized type >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_half) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_half) + sizeof(uint32_t) + QK5_0 / 2, "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_half) + sizeof(uint32_t) + QK5_1 / 2, "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR;
        ggml_half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1, "wrong q8_1 block size/padding");

//
// Super-block quantization structures
//

// Define QK4_NL for IQ4_NL type
#define QK4_NL 32

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Define quantized type traits    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
typedef struct {
    const char      * type_name;
    int               blck_size;
    size_t            type_size;
    bool              is_quantized;
    GGML_TYPE         vec_dot_type;
    int64_t           nrows; // number of rows to process simultaneously;
    bool              is_supported;
} ggml_type_traits_t;

// static ggml_type_traits_t typeTraits[GGML_TYPE_COUNT];
static const ggml_type_traits_t typeTraits[GGML_TYPE_COUNT] = {
    // GGML_TYPE_F32 = 0
    {"f32", 1, sizeof(float), false, GGML_TYPE_F32, 1, true},
    // GGML_TYPE_F16 = 1
    {"f16", 1, sizeof(ggml_fp16_t), false, GGML_TYPE_F16, 1, true},
    // GGML_TYPE_Q4_0 = 2
    // {"q4_0", QK4_0, sizeof(block_q4_0), true, GGML_TYPE_Q4_0, 1, false},
    // // GGML_TYPE_Q4_1 = 3
    // {"q4_1", QK4_1, sizeof(block_q4_1), true, GGML_TYPE_Q4_1, 1, false},
    // // GGML_TYPE_Q4_2 = 4 (deprecated)
    // {"DEPRECATED", 0, 0, false, GGML_TYPE_F32, 1, false},
    // // GGML_TYPE_Q4_3 = 5 (deprecated)
    // {"DEPRECATED", 0, 0, false, GGML_TYPE_F32, 1, false},
    // // GGML_TYPE_Q5_0 = 6
    // {"q5_0", QK5_0, sizeof(block_q8_0), true, GGML_TYPE_Q8_0, 1, false},
    // // GGML_TYPE_Q5_1 = 7
    // {"q5_1", QK5_1, sizeof(block_q5_1), true, GGML_TYPE_Q8_1, 1, false},
    // // GGML_TYPE_Q8_0 = 8
    {"q8_0", QK8_0, sizeof(block_q8_0), true, GGML_TYPE_Q8_0, 1, false},
    // // GGML_TYPE_Q8_1 = 9
    // {"q8_1", QK8_1, sizeof(block_q8_1), true, GGML_TYPE_Q8_1, 1, false},
    // // GGML_TYPE_Q2_K = 10
    // {"q2_K", QK_K, sizeof(block_q2_K), true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_Q3_K = 11
    // {"q3_K", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_Q4_K = 12
    // {"q4_K", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_Q5_K = 13
    // {"q5_K", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_Q6_K = 14
    // {"q6_K", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_Q8_K = 15
    // {"q8_K", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ2_XXS = 16
    // {"iq2_xxs", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ2_XS = 17
    // {"iq2_xs", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ3_XXS = 18
    // {"iq3_xxs", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ1_S = 19
    // {"iq1_s", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ4_NL = 20
    // {"iq4_nl", QK4_NL, 0, true, GGML_TYPE_Q8_0, 1, false},
    // // GGML_TYPE_IQ3_S = 21
    // {"iq3_s", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ2_S = 22
    // {"iq2_s", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_IQ4_XS = 23
    // {"iq4_xs", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false},
    // // GGML_TYPE_I8 = 24
    // {"i8", 1, sizeof(int8_t), false, GGML_TYPE_I8, 1, true},
    // // GGML_TYPE_I16 = 25
    // {"i16", 1, sizeof(int16_t), false, GGML_TYPE_I16, 1, true},
    // // GGML_TYPE_I32 = 26
    // {"i32", 1, sizeof(int32_t), false, GGML_TYPE_I32, 1, true},
    // // GGML_TYPE_I64 = 27
    // {"i64", 1, sizeof(int64_t), false, GGML_TYPE_I64, 1, true},
    // // GGML_TYPE_F64 = 28
    // {"f64", 1, sizeof(double), false, GGML_TYPE_F64, 1, true},
    // // GGML_TYPE_IQ1_M = 29
    // {"iq1_m", QK_K, 0, true, GGML_TYPE_Q8_K, 1, false}
};

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Common function  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Define quantized and de-quantized func  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Compute function  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

}
#endif //MINFER_GGML_QUANT_H
