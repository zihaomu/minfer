//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_BACKEND_H
#define MINFER_BACKEND_H

#include "non_copyable.h"
#include "minfer/tensor.h"
#include "string"

namespace minfer
{

class Backend : NonCopyable {
public:
    struct Config
    {
    };

    Backend(); // 创建Backend的具体限制，比如线程数，内存限制等。
    ~Backend();

    size_t memoryCost(); // 优化整个
    virtual Tensor createTensor(std::vector<int> shape, Tensor::DataType type) = 0; // 直接创建GPU Tensor

private:
    std::string name;
    Config config;
};

}


#endif //MINFER_BACKEND_H
