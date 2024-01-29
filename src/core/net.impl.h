//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_NET_IMPL_H
#define MINFER_NET_IMPL_H

// 先run起来，其他都是次要的。
#include "minfer/net.h"
#include "minfer/layer.h"
#include "minfer/tensor.h"

#include "runtime.h"

#include <map>
#include <vector>

namespace minfer
{

class Net::NetImpl
{
public:
    NetImpl();
    ~NetImpl();

    int createLayer(std::shared_ptr<LayerParams> param);

private:

    std::map<int, std::shared_ptr<Layer> > layers;    // layer 持有的内存只是一段param内存
    std::map<int, Tensor*> tensors; // Tensor是用于层之间的数据传输的，每个Tensor都有一个TensorDesc，用于描述Tensor的维度，数据类型等信息。
    std::map<int, int> tensorLayer; // tensorId -> layerId, every tensor belongs to a layer. and every layer has its output tensor.
    std::vector<int> inputTensorsId;
    std::vector<int> outputTensorsId;

    std::vector<int> inputLayers;
    Runtime* runtime; // 这里需要一个Runtime，用来管理所有的Device，以及所有的Tensor。
    // 需要一个全局单例模式去管理所有Device，然后再指向这个Device。
};

}

#endif //MINFER_NET_IMPL_H
