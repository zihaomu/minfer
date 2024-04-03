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

void Layer::init(const std::vector<Mat*> &, std::vector<Mat*> &)
{
    M_ERROR(NULL, "Not implementation at Layer::init!");
}

void Layer::forward(const std::vector<Mat*> &, std::vector<Mat*> &)
{
    M_ERROR(NULL, "Not implementation at Layer::forward!");
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

void Layer::getBasicInfo(const std::shared_ptr<LayerParams> param)
{
    layerId = param->layerId;
    layerType = param->type;
    layerName = param->name;
}

}