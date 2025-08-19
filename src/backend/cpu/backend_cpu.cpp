//
// Created by mzh on 2024/1/22.
//

#include "backend_cpu.h"

// all supported layer
#include "layer/add_layer.h"
#include "layer/input_layer.h"
#include "layer/output_layer.h"
#include "layer/attention_layer.h"
#include "layer/feed_forward.h"
#include "layer/embeding_layer.h"

namespace minfer
{

BackendCPU::BackendCPU()
{
    layerFactory = std::shared_ptr<LayerFactoryCPU>(new LayerFactoryCPU());
    memoryAllocatorCPUImpl = Allocator::AllocatorImpl::createDefault();
    memoryAllocatorCPU = std::shared_ptr<Allocator>(new Allocator(memoryAllocatorCPUImpl));
}

BackendCPU::~BackendCPU()
{
    memoryAllocatorCPU->release();
}

namespace details {

template<typename LayerClass>
std::shared_ptr<Layer> _layerDynamicRegister(std::shared_ptr<LayerParams> param)
{
    return std::shared_ptr<Layer>(LayerClass::create(param));
}

}

BackendCPU::LayerFactoryCPU::LayerFactoryCPU()
{
    this->registerAllLayer();
}

#define M_CPU_REGISTER_LAYER(type, class) \
    LayerFactory::registerLayer(type, minfer::details::_layerDynamicRegister<class>)

void BackendCPU::LayerFactoryCPU::registerAllLayer()
{
    // TODO Add more layer here.
    M_CPU_REGISTER_LAYER(LayerType::Add, AddLayer);
    M_CPU_REGISTER_LAYER(LayerType::Input, InputLayer);
    M_CPU_REGISTER_LAYER(LayerType::Output, OutputLayer);
    M_CPU_REGISTER_LAYER(LayerType::Attention, AttentionLayer);
    M_CPU_REGISTER_LAYER(LayerType::FFN, FeedForwardLayer);
    M_CPU_REGISTER_LAYER(LayerType::Embedding, EmbeddingLayer);
}

std::shared_ptr<Layer> BackendCPU::createLayer(std::shared_ptr<LayerParams> param)
{
    M_Assert(checkLayerSupported(param));
    return layerFactory->createLayerInstance(param);
}

int BackendCPU::allocMat(Mat* m)
{
    size_t totalMem = m->total() * DT_ELEM_SIZE(m->type());
    std::pair<void *, size_t> mPair = memoryAllocatorCPU->alloc(totalMem);
    m->data = (uchar *)mPair.first;
    return 0;
}

int BackendCPU::deallocMat(Mat *m)
{
    size_t totalMem = m->total() * DT_ELEM_SIZE(m->type());
    memoryAllocatorCPU->returnMemory(std::pair<void*, size_t>(m->data, totalMem));
    return 0;
}

}