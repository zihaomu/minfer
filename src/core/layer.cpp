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
//    layerName = param->name;
    layerType = param->type;
}

Layer::~Layer()
{

}

void Layer::init(const std::vector<Mat*> &, std::vector<Mat*> &)
{
    M_Error(Error::StsNotImplemented, "Not implementation at Layer::init!");
}

void Layer::finalize(const std::vector<Mat *> &, std::vector<Mat *> &)
{
    M_Error(Error::StsNotImplemented, "Not implementation at Layer::finalize!");
}

void Layer::forward(const std::vector<Mat*> &, std::vector<Mat*> &)
{
    M_Error(Error::StsNotImplemented, "Not implementation at Layer::forward!");
}

void Layer::forward(const std::vector<Mat*> & input, std::vector<Mat*> & output, int)
{
    this->forward(input, output);
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
    // layer id was set as layer create time.
    layerType = param->type;
}

}