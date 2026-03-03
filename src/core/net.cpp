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
    M_Assert(impl != nullptr);
    return impl->createLayer(param);
}

void Net::createNet(const std::vector<std::shared_ptr<LayerParams> > &netParams)
{
    M_Assert(impl != nullptr);
    return impl->createNet(netParams);
}

void Net::readNet(const std::string path, const std::string modelType)
{
    M_Assert(impl != nullptr);
    return impl->readNet(path, modelType);
}

void Net::setInput(const Mat input, const int mIndx)
{
    M_Assert(impl != nullptr);
    return impl->setInput(input, mIndx);
}

Mat Net::forward()
{
    M_Assert(impl != nullptr);
    return impl->forward();
}

void Net::init()
{
    M_Assert(impl != nullptr);
    return impl->init();
}

void Net::forward(Mat& out)
{
    M_Assert(impl != nullptr);
    return impl->forward(out);
}

void Net::generate(minfer::Mat &out)
{
    M_Assert(impl != nullptr);
    return impl->forward(out);
}

void Net::encode(const std::string text, std::vector<int> &out_ids)
{
    M_Assert(impl != nullptr);
    return impl->encode(text, out_ids);
}


void Net::decode(const std::vector<int> &out_ids, std::string &out_text)
{
    M_Assert(impl != nullptr);
    return impl->decode(out_ids, out_text);
}

Mat Net::prefill(const std::vector<int>& token_ids)
{
    M_Assert(impl != nullptr);
    return impl->prefill(token_ids);
}

Mat Net::step(int token_id)
{
    M_Assert(impl != nullptr);
    return impl->step(token_id);
}

void Net::resetKVCache()
{
    M_Assert(impl != nullptr);
    return impl->resetKVCache();
}

}