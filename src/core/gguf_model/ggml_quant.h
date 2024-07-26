//
// Created by mzh on 2024/5/23.
//

#ifndef MINFER_GGML_QUANT_H
#define MINFER_GGML_QUANT_H

#include <string>
#include <vector>
#include <map>
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

static const ggml_type_traits_t typeTraits[GGML_TYPE_COUNT] = {
        [GGML_TYPE_I8] = {
                .type_name                = "i8",
                .blck_size                = 1,
                .type_size                = sizeof(int8_t),
                .is_quantized             = false,
                .is_supported             = true,
        },
        [GGML_TYPE_I16] = {
                .type_name                = "i16",
                .blck_size                = 1,
                .type_size                = sizeof(int16_t),
                .is_quantized             = false,
                .is_supported             = true,
        },
        [GGML_TYPE_I32] = {
                .type_name                = "i32",
                .blck_size                = 1,
                .type_size                = sizeof(int32_t),
                .is_quantized             = false,
                .is_supported             = true,
        },
        [GGML_TYPE_I64] = {
                .type_name                = "i64",
                .blck_size                = 1,
                .type_size                = sizeof(int64_t),
                .is_quantized             = false,
                .is_supported             = true,
        },
        [GGML_TYPE_F64] = {
                .type_name                = "f64",
                .blck_size                = 1,
                .type_size                = sizeof(double),
                .is_quantized             = false,
                .nrows                    = 1,
                .is_supported             = true,
        },
        [GGML_TYPE_F32] = {
                .type_name                = "f32",
                .blck_size                = 1,
                .type_size                = sizeof(float ),
                .is_quantized             = false,
                .vec_dot_type             = GGML_TYPE_F32,
                .nrows                    = 1,
                .is_supported             = true,
        },
        [GGML_TYPE_F16] = {
                .type_name                = "f16",
                .blck_size                = 1,
                .type_size                = sizeof(ggml_fp16_t),
                .is_quantized             = false,
                .nrows                    = 1,
                .is_supported             = true,
        },
        [GGML_TYPE_Q4_0] = {
                .type_name                = "q4_0",
                .blck_size                = QK4_0,
                .is_quantized             = true,
                .is_supported             = false,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q4_1] = {
                .type_name                = "q4_1",
                .blck_size                = QK4_1,
                .is_quantized             = true,
                .is_supported             = false,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [4] = { // GGML_TYPE_Q4_2
                .type_name                = "DEPRECATED",
                .blck_size                = 0,
                .is_supported             = false,
                .is_quantized             = false,
                .nrows                    = 1,
        },
        [5] = { // GGML_TYPE_Q4_3
                .type_name                = "DEPRECATED",
                .blck_size                = 0,
                .is_supported             = false,
                .is_quantized             = false,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_0] = {
                .type_name                = "q5_0",
                .blck_size                = QK5_0,
                .is_supported             = false,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_1] = {
                .type_name                = "q5_1",
                .blck_size                = QK5_1,
                .is_supported             = false,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q8_0] = {
                .type_name                = "q8_0",
                .blck_size                = QK8_0,
                .is_quantized             = true,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q8_1] = {
                .type_name                = "q8_1",
                .blck_size                = QK8_1,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q2_K] = {
                .type_name                = "q2_K",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q3_K] = {
                .type_name                = "q3_K",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q4_K] = {
                .type_name                = "q4_K",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_K] = {
                .type_name                = "q5_K",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q6_K] = {
                .type_name                = "q6_K",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ2_XXS] = {
                .type_name                = "iq2_xxs",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ2_XS] = {
                .type_name                = "iq2_xs",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ3_XXS] = {
                .type_name                = "iq3_xxs",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ3_S] = {
                .type_name                = "iq3_s",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ2_S] = {
                .type_name                = "iq2_s",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ1_S] = {
                .type_name                = "iq1_s",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .nrows                    = 1,
        },
        [GGML_TYPE_IQ1_M] = {
                .type_name                = "iq1_m",
                .blck_size                = QK_K,
                .is_quantized             = true,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
//        [GGML_TYPE_IQ4_NL] = {
//                .type_name                = "iq4_nl",
//                .blck_size                = QK4_NL,
//                .is_quantized             = true,
//                .vec_dot_type             = GGML_TYPE_Q8_0,
//                .nrows                    = 1,
//        },
//        [GGML_TYPE_IQ4_XS] = {
//                .type_name                = "iq4_xs",
//#if QK_K == 64
//                .blck_size                = QK4_NL,
//#else
//                .blck_size                = QK_K,
//#endif
//                .is_quantized             = true,
//#if QK_K == 64
//                .vec_dot_type             = GGML_TYPE_Q8_0,
//#else
//                .vec_dot_type             = GGML_TYPE_Q8_K,
//#endif
//                .nrows                    = 1,
//        },
        [GGML_TYPE_Q8_K] = {
                .type_name                = "q8_K",
                .blck_size                = QK_K,
                .is_quantized             = true,
        }
};

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Common function  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Define quantized and de-quantized func  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Compute function  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

}
#endif //MINFER_GGML_QUANT_H
