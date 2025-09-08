//
// Created by mzh on 2024/8/5. 考虑把这个文件放到core中
//

#ifndef MINFER_UTILS_H
#define MINFER_UTILS_H

#include <cstdint> // uint32_t, uint64_t, etc.
#include "mat.h"

namespace minfer{

// taken from https://gist.github.com/zhuker/b4bd1fb306c7b04975b712c37c4c4075
float fp16_to_fp32(const uint16_t in);

// taken from https://gist.github.com/zhuker/b4bd1fb306c7b04975b712c37c4c4075
uint16_t fp32_to_fp16(const float in);

// convert the mat shape to char*, it is used to conveniently print mat dimension info.
std::string shape_to_str(const Mat& m);
std::string shape_to_str(const MatShape& shape);


// shape inferen of multi-dim gemm op
MatShape get_gemm_shape(const Mat& A, const Mat& B);
MatShape get_gemm_shape(const MatShape&A, const MatShape& B);

}

#endif //MINFER_UTILS_H
