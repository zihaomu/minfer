//
// Created by mzh on 2024/1/22.
//

#include "net.impl.h"
#include "gguf_model/gguf_loader.h"

namespace minfer
{

Net::NetImpl::NetImpl()
{
    if (runtime == nullptr)
    {
        runtime = Runtime::getRuntime();
    }
}

Net::NetImpl::~NetImpl()
{

}

void Net::NetImpl::readNet(const std::string path, const std::string modelType)
{
    // TODO Add model model type supported!
    std::map<int, std::shared_ptr<LayerParams> > netParams;
    M_ASSERT(modelType == "gguf" && "Only GGUF model has been supported!");

    readGGUF(path, netParams);

    createNet(netParams);
}

void Net::NetImpl::setInput(const Mat input, const int _mIndx)
{
    int mIndx = _mIndx;
    if (mIndx == -1)
    {
        M_ASSERT(inputMatId.size() == 1);
        mIndx = inputMatId[0];
    }

    auto it = std::find(inputMatId.begin(), inputMatId.end(), mIndx);
    const int index = it - inputMatId.begin();
    inputMatClone[index] = input.clone();

    // Update input Mat pointer.
    auto itLayerId = matId2layer.find(mIndx);
    M_ASSERT(itLayerId != matId2layer.end());

    auto& ld = lds[itLayerId->second];
    if (ld.layer->getType() == LayerType::Input)
    {
        M_ASSERT(ld.inputsIdx.size() == 1);

        ld.inputs[0] = &inputMatClone[index];

        auto itM = mats.find(index);
        M_ASSERT(itM != mats.end());

        itM->second = &inputMatClone[index];
    }
}

void Net::NetImpl::forward(Mat& out)
{
    out = this->forward();
}

Mat Net::NetImpl::forward()
{
    // TODO 完成这一步实现
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        it->layer->forward(it->inputs, it->outputs);
    }

    M_ASSERT(outputMatId.size() == 1);

    Mat* m = this->getMat(outputMatId[0]);
    M_ASSERT(m && "m can not be empty!");
    return *m;
}

void Net::NetImpl::init()
{
    // build layerId 2 Custom
    // TODO 优化下面代码，添加forward layer order部分优化，找到最佳的forward顺序。
    // 为什么需要调用两次这个代码？考虑到创建是的layerId和Params中的LayerId是不一样的，创建好的和实际运行速度最优的layer order也是不一样。
    // 所以这两个不一样需要调用两次这部分代码
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        std::vector<int> currCustom = {};

        // 如果只有一个输出，只需要找哪些层需要这个输出作为输入就行
        const std::vector<int>& outputIndex = it->outputsIdx;

        for (int i = 0; i < outputIndex.size(); i++)
        {
            int currOutIdx = outputIndex[i];
            for (auto it2 = lds.begin(); it2 != lds.end(); it2++)
            {
                const std::vector<int> inputIndex = it2->inputsIdx;
                auto itFind = std::find(inputIndex.begin(), inputIndex.end(), currOutIdx);
                if (itFind != inputIndex.end())
                {
                    currCustom.push_back(it2->layerId);
                }
            }
        }

        it->layerCustomers = currCustom;
    }

    // TODO 考虑将下面这段代码加入GPU
    // 建立Mat的使用表格，能达到最好的复用策略。
    // Mat表格指的是创建的Mat能在哪一层被释放
    std::map<int, int> matReleaseAtLayer; // mat Id to layerId
    for (auto it = mats.begin(); it != mats.end(); it++)
    {
        auto itLy = matId2layer.find(it->first);
        M_ASSERT(itLy != matId2layer.end());

        auto itLd = lds[itLy->second];

        if (itLd.layerCustomers.size() == 0)
            matReleaseAtLayer[it->first] = -1;
        else
        {
            int lastCustomId = -1;
            for (int i = 0; i < itLd.layerCustomers.size(); i++)
            {
                // ⚠️ 这里把最后一个包含此Mat的customId为释放的flag
                int customId = itLd.layerCustomers[i];
                if (customId > lastCustomId)
                {
                    // 查找这个custome layer的input是否包含这个Mat
                    auto ld = lds[customId];

                    const std::vector<int>& inputsIds = ld.inputsIdx;
                    if (std::find(inputsIds.begin(), inputsIds.end(), it->first) != inputsIds.end())
                    {
                        lastCustomId = customId;
                    }
                }
            }

            // 找到这个mat能在那一层运行完之后被释放
            matReleaseAtLayer[it->first] = lastCustomId;
        }
    }

    // 调用runtime 分配和释放内存
    for (auto it = lds.begin(); it != lds.end(); it++)
    {
        it->layer->init(it->inputs, it->outputs); // 计算shape

        // 分配内存
        for (int i = 0; i < it->outputsIdx.size(); i++)
        {
            runtime->allocMat(it->outputs[i]);
        }

        // 释放不用的资源，查找释放flag，确定是否在当前layerId释放
        for (int i = 0; i < it->outputsIdx.size(); i++)
        {
            int outId = it->outputsIdx[i];
            auto it2 = matReleaseAtLayer.find(outId);

            M_ASSERT(it2 != matReleaseAtLayer.end());

            if (it2->second == it->layerId)
            {
                runtime->deallocMat(it->outputs[i]); // 回收当前资源
            }
        }
    }
}

void Net::NetImpl::createLayerRecurve(int layerIdx, std::vector<int>& isLayerCreated, const std::map<int,
        std::vector<int> >& layer2Custom,  const std::map<int, std::shared_ptr<LayerParams> >& allLayerParams)
{
    if(isLayerCreated[layerIdx])
    {
        return;
    }

    // create layer's parents first
    auto it = layer2Custom.find(layerIdx);
    M_ASSERT(it != layer2Custom.end());

    for (int i = 0; i < it->second.size(); i++)
    {
        createLayerRecurve(it->second[i], isLayerCreated, layer2Custom, allLayerParams);
    }

    auto it2 = allLayerParams.find(layerIdx);
    M_ASSERT(it2 != allLayerParams.end());

    if (createLayer(it2->second) >= 0)
    {
        isLayerCreated[layerIdx] = 1;
    }
}

// 此函数保证在 allLayerParams乱序情况下，仍然能够让模型从input层一层层创建，从而让后面层的创建滞后于前面的层。
// 此部分代码有待测试
void Net::NetImpl::createNet(const std::map<int, std::shared_ptr<LayerParams> >& allLayerParams)
{
    // find all net output node

    // 建立网格，找到谁是谁的消费者
    std::map<int, std::vector<int> > layer2Custom; // 建立layerid 到消费者的网格
    std::vector<int> outLayerIndex;
    for (auto it = allLayerParams.begin(); it != allLayerParams.end(); it++)
    {
        std::vector<int> currCustom = {};

        // 如果只有一个输出，只需要找哪些层需要这个输出作为输入就行
        for (int i = 0; i < it->second->outputIndex.size(); i++)
        {
            int currOutIdx = it->second->outputIndex[i];
            for (auto it2 = allLayerParams.begin(); it2 != allLayerParams.end(); it2++)
            {
                auto itFind = std::find(it2->second->inputIndex.begin(), it2->second->inputIndex.end(), currOutIdx);
                if (itFind != it2->second->inputIndex.end())
                {
                    currCustom.push_back(it2->first);
                }
            }
        }

        layer2Custom[it->first] = currCustom;
        if (currCustom.size() == 0)
        {
            outLayerIndex.push_back(it->first);
        }
    }

    std::vector<int> isLayerCreated(allLayerParams.size(), 0);
    // 递归的调用createLayerParents，建立是否创建表格。
    for (int i = 0; i < outLayerIndex.size(); i++)
    {
        createLayerRecurve(outLayerIndex[i], isLayerCreated, layer2Custom, allLayerParams);
    }
}

int Net::NetImpl::createLayer(std::shared_ptr<LayerParams> param)
{
    AutoLock lk(mutex);
    // TODO 对inputlayer和outputlayer的特殊处理

    // Check if the input layer has been created.
    int inputSize = param->inputIndex.size();
    int outputSize = param->outputIndex.size();

    LayerData ld = {};
    int layerId = lds.size();
    std::shared_ptr<Layer> layer = runtime->createLayer(param);

    // 对输入输出对特殊处理
    // 输入将会在setinput中进行初始化。
    if (param->type == LayerType::Input)
    {
        inputMatClone.push_back(Mat());
        inputMatId.push_back(param->inputIndex[0]);
        M_ASSERT(param->inputIndex.size() == 1);
        mats[param->inputIndex[0]] = nullptr;
        inputLayers.push_back(layerId);
        matId2layer[param->inputIndex[0]] = layerId;
    }
    else if (param->type == LayerType::Output)
    {
        M_ASSERT(param->outputIndex.size() == 1);
        outputMatId.push_back(param->outputIndex[0]);
        outputLayers.push_back(layerId);
    }

    // 每一个层只管理自己的outputMat，而inputMat是由上面传下来的
    std::vector<Mat*> inps(inputSize, nullptr);
    for (int i = 0; i < inputSize; ++i)
    {
        int inputId = param->inputIndex[i];
        Mat* m = getMat(inputId);
        if (!m)
        {
            // The input Mat has not been created.
            M_ERROR("The input Mat has not been created!");
        }
        inps[i] = m;
    }

    layer->setId(layerId);

    std::vector<Mat*> outs(outputSize, nullptr);
    for (int i = 0; i < outputSize; ++i)
    {
        int outputMatId = param->outputIndex[i];
        outs[i] = new Mat();
        mats[outputMatId] = outs[i];
        matId2layer[outputMatId] = layerId;
    }

    ld.layerId = layerId;
    ld.layer = layer;
    ld.inputs = inps;
    ld.inputsIdx = param->inputIndex;
    ld.outputs = outs;
    ld.outputsIdx = param->outputIndex;

    lds.push_back(ld);
    // Create the layer.
    return layerId;
}

Mat *Net::NetImpl::getMat(const int matIdx)
{
    auto it = mats.find(matIdx);

    if (it != mats.end())
    {
        return it->second;
    }
    else
    {
        return nullptr;
    }
}

void Net::NetImpl::getMats(const std::vector<int> matsIdx, std::vector<Mat *> &mats)
{
    mats.clear();
    mats.resize(matsIdx.size(), nullptr);

    for (int i = 0; i < matsIdx.size(); i++)
    {
        mats[i] = getMat(matsIdx[i]);
    }
}

}