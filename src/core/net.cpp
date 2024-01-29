//
// Created by mzh on 2024/1/22.
//

#include "minfer/net.h"
#include "net.impl.h"

namespace minfer
{

Net::Net()
{
    impl = new NetImpl();
}

int Net::createLayer(std::shared_ptr<LayerParams> param)
{
    M_ASSERT(impl != nullptr);
    return impl->createLayer(param);
}

}