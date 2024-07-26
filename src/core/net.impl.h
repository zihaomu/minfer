//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_NET_IMPL_H
#define MINFER_NET_IMPL_H

#include "minfer/net.h"
#include "minfer/layer.h"
#include "minfer/mat.h"

#include "runtime.h"

#include <map>
#include <vector>

namespace minfer
{

struct LayerData
{
    int layerId = -1;
    std::shared_ptr<Layer> layer;
    std::vector<Mat*> inputs;
    std::vector<int> inputsIdx;
    std::vector<Mat*> outputs;
    std::vector<int> outputsIdx;
    std::vector<int> layerCustomers;
};

class Net::NetImpl
{
public:
    NetImpl();
    ~NetImpl();

    // 内部需要解析多个模型结构
    void readNet(const std::string path, const std::string modelType);

    void setInput(const Mat input, const int mIndx);

    // 此处转换出去的Mat必须是CPU内存
    void forward(Mat& out);

    Mat forward();

    int createLayer(std::shared_ptr<LayerParams> param);

    void createNet(const std::vector<std::shared_ptr<LayerParams> >& allLayerParams);

    void init(); // 初始化之后，调用系统中已经注册好的全局Backend变量。

private:
    void createLayerRecurve(int layerIdx, std::vector<int>& isLayerCreated, const std::map<int,
            std::vector<int> >& layer2Parent, const std::vector<std::shared_ptr<LayerParams> >& allLayerParams);

    Mat* getMat(const int matIdx);
    void getMats(const std::vector<int> matsIdx, std::vector<Mat*>& mats);

    Mutex mutex;
    std::vector<LayerData> lds;     // contains all layer data inform
    std::map<int, Mat*> mats;       // Mat是用于层之间的数据传输的，这里建立MatId和Mat的对应关系
    std::map<int, int> matId2layer; // matId -> layerId, 每个Mat都属于一个层，一个层可以拥有多个Mat。实际这里的Mat都是对应层的output Mat。

    std::vector<Mat> inputMatClone; // 包含Net输入Mat实体
    std::vector<int> inputMatId;    // 包含模型的输入Mat id
    std::vector<int> outputMatId;   // 包含模型的输出Mat
    std::vector<int> inputLayers;   // 存储input layer id
    std::vector<int> outputLayers;  // 存储output layer id

    // Runtime 相当于全局的资源管理器，其中包含多个。
    // TODO, 一个Net应该包含多个runtime或者backend，互相连接
    Runtime* runtime; // 这里需要一个Runtime，用来管理所有的Device，以及所有的Tensor。// 是不是用Backend就可以，还是需要再封一层？
    // 需要一个全局单例模式去管理所有Device，然后再指向这个Device。
};

}

#endif //MINFER_NET_IMPL_H
