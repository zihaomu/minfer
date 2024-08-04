//
// Created by mzh on 2024/1/22.
//

#include "minfer/tensor.h"
#include "memory_utils.h"
#include "minfer/system.h"

namespace minfer
{

Tensor::Tensor()
: data(nullptr), dType(DT_INVALID), mType(HOST_MEMORY)
{
    extraInfo = std::shared_ptr<TensorExtraInfo>();
}

size_t Tensor::total(int start, int end)
{
    if (shape.empty())
        return 0;

    int dims = (int)shape.size();

    if (start == -1) start = 0;
    if (end == -1) end = dims;

    size_t elems = 1;
    for (int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

Tensor::Tensor(const std::vector<int> &_shape, minfer::Tensor::DataType type, void *_data)
: shape(_shape), data(_data), dType(type), mType(HOST_MEMORY)
{
    M_Assert(type == DT_FLOAT); // TODO to add more DT support for Tensor

    if (data)
    {
        // do nothing.
    }
    else
    {
        size_t memSize = total() * sizeof (float );
        data = MMemoryAllocAlign(memSize, M_MEMORY_ALIGN_DEFAULT);
    }
    extraInfo = std::shared_ptr<TensorExtraInfo>();
}

Tensor::Tensor(const Tensor &tensor)
{

}

Tensor &Tensor::operator=(const minfer::Tensor &t)
{

}

void Tensor::print()
{
    printShape();
    size_t totalSize = total();

    float* fp = (float *)data;
    for (int i = 0; i < totalSize; i++)
    {
        printf(", %f", fp[i]);
    }
    printf("\n");
}

void Tensor::printShape()
{
    if (dType == DT_FLOAT)
        printf("T type is float.");

    if (shape.empty())
    {
        printf("T shape = [] \n");
    }
    else if (shape.size() == 1)
    {
        printf("T shape = [%d] \n", shape[0]);
    }
    else
    {
        printf("T shape = [%d x", shape[0]);
        for (int i = 1; i < shape.size(); i++)
        {
            printf(" %d x",shape[i]);
        }
        printf("]\n");
    }
}

Tensor::~Tensor()
{

}

void Tensor::copyDeviceToHost()
{
    M_ERROR("Not implemented!");
}

void Tensor::copyHostToDevice()
{
    M_ERROR("Not implemented!");
}

}