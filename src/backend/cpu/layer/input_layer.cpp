//
// Created by mzh on 2024/3/28.
//

#include "input_layer.h"

namespace minfer {

InputLayer::~InputLayer()
{

}

void InputLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    // 设置同样的shape
    output[0]->setSize(*input[0]);
}

void InputLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    (*input[0]).print();
    if (output[0]->data != input[0]->data)
    {
        size_t totalSize = input[0]->total() * DT_ELEM_SIZE(input[0]->type());
        memcpy(output[0]->data, input[0]->data, totalSize);
    }
    else
    {
        std::cout<<"WARNING: output[0]->data == input[0]->data at InputLayer::forward"<<std::endl;
    }
    (*output[0]).print();
}

InputLayer::InputLayer(const std::shared_ptr<LayerParams> param)
{
    M_Assert(param->type == LayerType::Input);
    getBasicInfo(param);
}

std::shared_ptr<InputLayer> InputLayer::create(const std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<InputLayer>(new InputLayer(param));
}

}