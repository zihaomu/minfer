//
// Created by mzh on 2024/3/27.
//

#ifndef MINFER_BASIC_OP_H
#define MINFER_BASIC_OP_H

#include "mat.h"

namespace minfer
{

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

#define MAT_AUG_OPERATOR1(op, cvop) \

#define MAT_AUG_OPERATOR(op, cvop) \
static inline Mat &operator op (Mat& a, const Mat& b) {cvop; return a;} \
static inline const Mat &operator op (const Mat& a, const Mat& b) {cvop; return a;}

MAT_AUG_OPERATOR(+=, add(a, b, (Mat &) a))
MAT_AUG_OPERATOR(-=, subtract(a, b, (Mat &) a))
MAT_AUG_OPERATOR(*=, multiply(a, b, (Mat &) a))
MAT_AUG_OPERATOR(/=, divide(a, b, (Mat &) a))

}
#endif //MINFER_BASIC_OP_H
