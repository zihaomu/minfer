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

Net::~Net()
{
}

int Net::createLayer(std::shared_ptr<LayerParams> param)
{
    M_ASSERT(impl != nullptr);
    return impl->createLayer(param);
}

void Net::createNet(const std::vector<std::shared_ptr<LayerParams> > &netParams)
{
    M_ASSERT(impl != nullptr);
    return impl->createNet(netParams);
}

void Net::readNet(const std::string path, const std::string modelType)
{
    M_ASSERT(impl != nullptr);
    return impl->readNet(path, modelType);
}

void Net::setInput(const Mat input, const int mIndx)
{
    M_ASSERT(impl != nullptr);
    return impl->setInput(input, mIndx);
}

Mat Net::forward()
{
    M_ASSERT(impl != nullptr);
    return impl->forward();
}

void Net::init()
{
    M_ASSERT(impl != nullptr);
    return impl->init();
}

void Net::forward(Mat& out)
{
    M_ASSERT(impl != nullptr);
    return impl->forward(out);
}

//void Net::forward(std::vector<Mat>& outs, const std::vector<std::string> names)
//{
//    M_ASSERT(impl != nullptr);
//    return impl->forward(outs, names);
//}
}