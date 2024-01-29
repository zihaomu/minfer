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

int Runtime::createLayer(std::shared_ptr<LayerParams> param)
{
    // check if the backend is supported, then check if the layer is supported by the backend. 
    // the create the layer by the specific backend.

    return 0;
}


}