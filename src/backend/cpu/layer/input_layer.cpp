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
    if (input[0]->empty())
        M_Error(Error::Code::StsBadArg, "The input Mat at InputLayer::init is empty! Please set input data before call init()!");
    // 设置同样的shape
    output[0]->setSize(*input[0]);
}

void InputLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    // (*input[0]).print();
    if (output[0]->data != input[0]->data || output[0]->empty())
    {
        // size_t totalSize = input[0]->total() * DT_ELEM_SIZE(input[0]->type());
        *output[0] = input[0]->clone();
    }
    else
    {
        std::cout<<"WARNING: output[0]->data == input[0]->data at InputLayer::forward"<<std::endl;
    }
    // (*output[0]).print();
}

InputLayer::InputLayer(const std::shared_ptr<LayerParams> param)
{
    layerNamePrefix = "InputLayer_";
    M_Assert(param->type == LayerType::Input);
    getBasicInfo(param);
}

std::shared_ptr<InputLayer> InputLayer::create(const std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<InputLayer>(new InputLayer(param));
}

}