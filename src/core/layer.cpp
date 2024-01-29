//
// Created by mzh on 2024/1/26.
//

#include "minfer/layer.h"

namespace minfer
{

Layer::Layer()
{
    layerId = -1;
    layerName = "";
    layerType = UnSupported;
}

Layer::Layer(std::shared_ptr<LayerParams> param)
{
    layerId = -1;
    layerName = param->name;
    layerType = param->type;
}

Layer::~Layer()
{

}

void Layer::init(const std::vector<Tensor> &, std::vector<Tensor> &)
{

}

void Layer::forward(const std::vector<Tensor> &, std::vector<Tensor> &)
{

}

int Layer::getId()
{
    return layerId;
}

std::string Layer::getName()
{
    return layerName;
}

LayerType Layer::getType()
{
    return layerType;
}

void Layer::setId(int id)
{
    layerId = id;
}

}