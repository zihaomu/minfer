//
// Created by mzh on 2024/1/22.
//

#include "backend_cpu.h"
#include "layer/add_layer.h"

namespace minfer
{

BackendCPU::BackendCPU()
{
    layerFactory = std::shared_ptr<LayerFactoryCPU>();
}

BackendCPU::~BackendCPU()
{

}

namespace details {

template<typename LayerClass>
std::shared_ptr<Layer> _layerDynamicRegister(std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<Layer>(LayerClass::create(param));
}

}

#define M_CPU_REGISTER_LAYER(type, class) \
    LayerFactory::registerLayer(type, minfer::details::_layerDynamicRegister<class>)

void BackendCPU::LayerFactoryCPU::registerAllLayer()
{
    // TODO Add more layer here.
    M_CPU_REGISTER_LAYER(LayerType::Add, AddLayer);
}

std::shared_ptr<Layer> BackendCPU::createLayer(std::shared_ptr<LayerParams> param)
{
    M_ASSERT(checkLayerSupported(param->type, param))

    return layerFactory->createLayerInstance(param);
}

int BackendCPU::allocTensorMemory(minfer::Tensor *tensor, std::vector<int> shape, Tensor::DataType type)
{
    return 0;
}


}