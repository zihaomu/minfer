//
// Created by mzh on 2024/1/22.
//

#include "backend.h"

namespace minfer {

Backend::LayerFactory::LayerFactory()
{
}

Backend::LayerFactory::~LayerFactory()
{
}

void Backend::LayerFactory::registerAllLayer()
{
    M_Error(NULL, "registerAllLayer is empty, please override it!");
}

void Backend::LayerFactory::registerLayer(LayerType type, Backend::LayerFactory::Constructor constructor)
{
    AutoLock lk(mutex);

    // check if the layer has been registered
    LayerFactoryMap::iterator it = layerMap.find(type);

    if (it != layerMap.end())
    {
        M_Error_(NULL, ("registerLayer failed! Layer type %d already was registered!", (int)type));
    }

    layerMap.insert(std::make_pair(type, constructor));
}

std::shared_ptr<Layer> Backend::LayerFactory::createLayerInstance(std::shared_ptr<LayerParams> param)
{
    AutoLock lk(mutex);

    LayerFactoryMap::iterator it = layerMap.find(param->type);

    if (it == layerMap.end())
    {
        M_Error_(NULL, ("createLayerInstance failed! Layer type %d was not registered!", (int)param->type));
        return std::shared_ptr<Layer>();
    }

    return it->second(param);
}

bool Backend::LayerFactory::checkLayerSupported(std::shared_ptr<LayerParams> param)
{
    auto it = layerMap.find(param->type);
    return !(it == layerMap.end());
}

bool Backend::checkLayerSupported(std::shared_ptr<LayerParams> param)
{
    M_Assert(layerFactory != nullptr);

    return layerFactory->checkLayerSupported(param);
}

size_t Backend::getAllMemory()
{
    return allocMemSize;
}

std::string Backend::getName()
{
    return name;
}

Backend::Backend(std::string backendName)
:name(backendName), allocMemSize(0)
{
    layerFactory = nullptr;
}

Backend::~Backend()
{

}

}