//
// Created by mzh on 2024/1/22.
//

#include "net.impl.h"

namespace minfer
{

Net::NetImpl::NetImpl()
{
    if (runtime == nullptr)
    {
        runtime = Runtime::getRuntime();
    }
}

Net::NetImpl::~NetImpl()
{

}

int Net::NetImpl::createLayer(std::shared_ptr<LayerParams> param)
{
    // Check if the input layer has been created.
    int inputTensorSize = param->inputTensorIndex.size();
    int outputTensorSize = param->outputTensorIndex.size();

    for (int i = 0; i < inputTensorSize; ++i)
    {
        int inputTensorId = param->inputTensorIndex[i];

        if (tensors.find(inputTensorId) == tensors.end())
        {
            // The input tensor has not been created.
            M_ERROR("The input tensor has not been created!");
        }
    }

    for (int i = 0; i < outputTensorSize; ++i)
    {
        int outputTensorId = param->outputTensorIndex[i];
        Tensor* tensor = new Tensor();
        tensors[outputTensorId] = tensor;
    }

    // Create the layer.
    return runtime->createLayer(param);
}

}