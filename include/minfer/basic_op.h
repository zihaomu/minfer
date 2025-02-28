//
// Created by mzh on 2024/3/27.
//

#ifndef MINFER_BASIC_OP_H
#define MINFER_BASIC_OP_H

#include "mat.h"

namespace minfer
{

// BinaryOp
enum BinaryOp
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
    MOD,  // Integer Mod. Reminder's sign = Divisor's sign.
    MUL,
    SUB,
    ADD,
    DIV,

// TODO support the following op.
//        SUM,
//        FMOD, // Floating-point Mod. Reminder's sign = Dividend's sign.
//        MAX,
//        MEAN,
//        MIN,
};

void binaryFunc(BinaryOp op, const Mat& a, const Mat& b, Mat& c);

// a + b = c
void add(const Mat& a, const Mat& b, Mat& c);

// a * alpha + b * beta = c
void addWeighted(const Mat& a, double alpha, const Mat& b, double beta, Mat& c);

// -a = c
void subtract(const Mat& a, Mat& c);

// a - b = c
void subtract(const Mat& a, const Mat& b, Mat& c);

// a * b = c
void multiply(const Mat& a, const Mat& b, Mat& c);

// a / b = c
void divide(const Mat& a, const Mat& b, Mat& c);

void compare(const Mat& a, const Mat& b, Mat& c, int op);

// Transpose Mat last two dimension, if the Mat is one dimension, add the axis to the shape.
Mat transpose(const Mat& input);

// transpose Mat according to the input mat and the given new order.
Mat transposeND(const Mat& input, const std::vector<int> order);

enum NormType
{
    NORM_L1 = 1,
    NORM_L2 = 2,
    NORM_INF = 3,
};

// compute the norm of the Mat.
double norm(const Mat& a, int normType);
double norm(const Mat& a, const Mat& b, int normType);

// reshape Mat according to the given shape.
void reshape(const Mat& input, const std::vector<int>& shape, Mat& out);

#define MAT_AUG_OPERATOR1(op, cvop) \

#define MAT_AUG_OPERATOR(op, cvop) \
static inline Mat &operator op (Mat& a, const Mat& b) {cvop; \
    a.print(10); \
    b.print(10); \
    return a;} \
static inline const Mat &operator op (const Mat& a, const Mat& b) {cvop; return a;}

MAT_AUG_OPERATOR(+=, add(a, b, (Mat &) a))
MAT_AUG_OPERATOR(-=, subtract(a, b, (Mat &) a))
MAT_AUG_OPERATOR(*=, multiply(a, b, (Mat &) a))
MAT_AUG_OPERATOR(/=, divide(a, b, (Mat &) a))

}
#endif //MINFER_BASIC_OP_H
