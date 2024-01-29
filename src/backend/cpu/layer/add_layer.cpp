//
// Created by mzh on 2024/1/24.
//

#include "add_layer.h"

namespace minfer {

AddLayer::AddLayer(std::shared_ptr<LayerParams> param)
{
    layerName = param->name;
    layerId = param->layerId;

    layerType = param->type;
    M_ASSERT(layerType == LayerType::Add);
}

AddLayer::~AddLayer()
{

}

void AddLayer::init(const std::vector<Tensor> &input, std::vector<Tensor> &output)
{
    // pre check
    inputNum = input.size();

    M_ASSERT(inputNum == 2);
    M_ASSERT(output.size() == 1);
}

void AddLayer::forward(const std::vector<Tensor> &input, std::vector<Tensor> &output)
{
    // TODO finish the following code!
    ; // do nothing
//    output[0] = add(input[0],input[1]);
}

std::shared_ptr<AddLayer> AddLayer::create(std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<AddLayer>(new AddLayer(param));
}
}