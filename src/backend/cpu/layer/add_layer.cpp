//
// Created by mzh on 2024/1/24.
//

#include "add_layer.h"

namespace minfer {

AddLayer::AddLayer(const std::shared_ptr<LayerParams> param)
{
    M_ASSERT(param->type == LayerType::Add);
    getBasicInfo(param);
}

AddLayer::~AddLayer()
{

}

void AddLayer::init(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    // pre check
    inputNum = input.size();

    M_ASSERT(inputNum == 2);
    M_ASSERT(output.size() == 1);

    // 设置同样的shape
    output[0]->setSize(*input[0]);
}

void AddLayer::forward(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    // TODO finish the following code!

    (*input[0]).print();
    (*input[1]).print();
    *output[0] = *input[0] + *input[1];
    (*output[0]).print();
}

std::shared_ptr<AddLayer> AddLayer::create(const std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<AddLayer>(new AddLayer(param));
}

}