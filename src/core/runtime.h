
#ifndef MINFER_RUNTIME_H
#define MINFER_RUNTIME_H

#include <memory>
#include "minfer/layer.h"

// Add all backend here
#include "backend/cpu/backend_cpu.h"

namespace minfer
{

// Runtime is a singleton class.
class Runtime
{
public:
    ~Runtime();
    
    // Runtime is a singleton class.
    static Runtime* getRuntime();

    // Runtime is a singleton class.
    // This function will release all the Tensor that keeped in net.
    static void release();

    // Create layer, and return layerId.
    std::shared_ptr<Layer> createLayer(std::shared_ptr<LayerParams> param);

    //TODO：怎么加入GPU内存分配？以及GPU内存复用？
    int allocMat(Mat* m);

    int deallocMat(Mat* m);

private:
    Runtime(); //Runtime 管理所有的Backend
    std::shared_ptr<BackendCPU> backendCPU = nullptr; // cpu backend
//    std::shared_ptr<BackendGPU> backendGPU = nullptr; // cpu backend
};

static Runtime* runtime = nullptr;

} // namespace minfer


#endif //MINFER_RUNTIME_H
