//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_LAYER_H
#define MINFER_LAYER_H

#include <string>
#include <vector>
#include "mat.h"

namespace minfer
{

// TODO add all the layer type.
enum LayerType {
    UnSupported = -1,
    Input = 0, // input layer only has the output Mat
    Output = 1, // output layer only has the input Mat
    LayerNormal,
    Attention,
    FFN,
    CONVOLUTION,
    Add,
};

// 这里是否应当包含模型参数信息？
// 模型初始化中，应当保持
// 每一个层，如果有额外的需要，都应该重写这部分，从而添加额外的参数信息。
class LayerParams
{
public:
    LayerParams()
    :layerId(-1), name(), type(UnSupported), inputIndex({}), outputIndex({})
    {}
    LayerParams(int _layerId, std::string _name, LayerType _type, std::vector<int> _inputIndex, std::vector<int> _outputIndex)
    :layerId(_layerId), name(_name), type(_type), inputIndex(_inputIndex), outputIndex(_outputIndex)
    {}
    int layerId = -1;
    std::string name;
    LayerType type;
    std::vector<int> inputIndex;
    std::vector<int> outputIndex;
};

// layer 层抽象
class Layer {
public:
    explicit Layer(const std::shared_ptr<LayerParams> param);

    Layer();

    ~Layer();

    // 此处的Tensor囊括了GPU的内存数据。

    // 初始化，根据输入Mat的shape，完成输出shape的计算，以及一些其他的初始化
    virtual void init(const std::vector<Mat*>& input, std::vector<Mat*>& output);

    // and the forward can be run several times
    virtual void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output);

    void setId(int id);

    int getId();

    std::string getName();

    LayerType getType();

protected:
    void getBasicInfo(const std::shared_ptr<LayerParams> param);
    int layerId; // layer id 是layer在Net中前后顺序的序号，保存在Net的layerList中
    std::string layerName;
    LayerType layerType; // layerType是层类型
};

}

#endif //MINFER_LAYER_H
