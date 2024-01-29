//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_BACKEND_CPU_H
#define MINFER_BACKEND_CPU_H

#include "core/backend.h"

namespace minfer {


class BackendCPU : public Backend
{
public:
    class LayerFactoryCPU : public LayerFactory
    {
    public:
        LayerFactoryCPU() = default;
        void registerAllLayer() override;

    private:

    };

    BackendCPU();
    ~BackendCPU();

    std::shared_ptr<Layer> createLayer(std::shared_ptr<LayerParams> param) override;
    // alloc specific memory for tensor. It will use some reuse strategy.
    int allocTensorMemory(Tensor* tensor, std::vector<int> shape, Tensor::DataType type) override;

private:

};

}


#endif //MINFER_BACKEND_CPU_H
