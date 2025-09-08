#include "runtime.h"

namespace minfer
{

Runtime::Runtime()
{
    // TODO: register all backends here.
    backendCPU = std::make_shared<BackendCPU>();
}

Runtime::~Runtime()
{

}

Runtime* Runtime::getRuntime()
{
    if (runtime == nullptr)
    {
        runtime = new Runtime();
    }
    return runtime;
}

void Runtime::release()
{
    if (runtime != nullptr)
    {
        delete runtime;
        runtime = nullptr;
    }

    // TODO release all Backend.
}

std::shared_ptr<Layer> Runtime::createLayer(std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<Layer> layer = {};
    // check if the backend is supported, then check if the layer is supported by the backend. 
    // the create the layer by the specific backend.
    if (backendCPU->checkLayerSupported(param))
    {
        layer = backendCPU->createLayer(param);
    }
    else
    {
        M_Error_(Error::StsBadFunc, ("createLayerInstance failed! Layer type %d was not registered! ! \n", (int)param->type));
    }

    return layer;
}

// TODO 增加GPU内存分配和回收的机制，需要制定Mat内存机制，确定Mat的内存flag，在分配时启动不同的backend进行分配。
int Runtime::allocMat(Mat *m)
{
    return backendCPU->allocMat(m);
}

int Runtime::deallocMat(Mat *m)
{
    return backendCPU->deallocMat(m);
}

}