//
// Created by mzh on 2024/3/28.
//

#include "output_layer.h"

namespace minfer {

void OutputLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    // 设置同样的shape
    output[0]->setSize(*input[0]);
}

void OutputLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    if (output[0]->data != input[0]->data)
    {
        size_t totalSize = input[0]->total() * DT_ELEM_SIZE(input[0]->type());
        memcpy(output[0]->data, input[0]->data, totalSize);
    }
    else
    {
        std::cout<<"WARNING: output[0]->data == input[0]->data at OutputLayer::forward"<<std::endl;
    }
}

OutputLayer::~OutputLayer()
{

}

OutputLayer::OutputLayer(const std::shared_ptr<LayerParams> param)
{
    layerNamePrefix = "OutputLayer_";
    M_Assert(param->type == LayerType::Output);
    getBasicInfo(param);
}

std::shared_ptr<OutputLayer> OutputLayer::create(const std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<OutputLayer>(new OutputLayer(param));
}

}