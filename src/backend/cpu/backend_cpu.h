//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_BACKEND_CPU_H
#define MINFER_BACKEND_CPU_H

#include "core/backend.h"
#include "core/allocator.h"

namespace minfer {


class BackendCPU : public Backend
{
public:
    class LayerFactoryCPU : public LayerFactory
    {
    public:
        LayerFactoryCPU();
        void registerAllLayer() override;

    private:

    };

    BackendCPU();
    ~BackendCPU();

    std::shared_ptr<Layer> createLayer(std::shared_ptr<LayerParams> param) override;

    // alloc specific memory for tensor. It will use some reuse strategy.
    int allocMat(Mat* m) override;

    int deallocMat(Mat* m) override;

private:
    std::shared_ptr<Allocator::AllocatorImpl> memoryAllocatorCPUImpl;
    std::shared_ptr<Allocator> memoryAllocatorCPU;

};

}


#endif //MINFER_BACKEND_CPU_H
