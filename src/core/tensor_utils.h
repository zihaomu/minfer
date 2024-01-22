//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_TENSOR_UTILS_H
#define MINFER_TENSOR_UTILS_H

#include "minfer/tensor.h"
#include "backend.h"

namespace minfer {

struct QuantAttr {
    float scale;
    float zero = 0.0f;
    float min  = -127.0f;
    float max  = 127.0f;
};

struct Tensor::TensorExtraInfo
{
    // Tensor Quant Attribute
    std::shared_ptr<QuantAttr> quantAttr;
    Backend* backend; // 对于GPU Tensor，其应当指向一个固定的backend。

    // Other platform information
};

}


#endif //MINFER_TENSOR_UTILS_H
