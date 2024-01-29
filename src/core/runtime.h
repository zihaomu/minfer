
#ifndef MINFER_RUNTIME_H
#define MINFER_RUNTIME_H

#include <memory>
#include "backend/cpu/backend_cpu.h"
#include "minfer/layer.h"

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
    int createLayer(std::shared_ptr<LayerParams> param);

private:
    Runtime(); //Runtime 管理所有的Backend
    std::shared_ptr<BackendCPU> backendCPU; // cpu backend
//    std::shared_ptr<BackendGPU> backendGPU; // cpu backend
};

static Runtime* runtime = nullptr;

} // namespace minfer


#endif //MINFER_RUNTIME_H
