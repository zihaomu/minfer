//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_LAYER_H
#define MINFER_LAYER_H

#include <string>
#include <vector>
#include "tensor.h"

namespace minfer
{

// TODO add all the layer type.
enum LayerType {
    UnSupported = -1,
    Input = 0, // input layer only has the output tensor
    LayerNormal,
    Attention,
    FFN,
    CONVOLUTION,
    Add,
};

class LayerParams
{
public:
    int layerId = -1;
    std::string name;
    LayerType type;
    std::vector<int> inputTensorIndex;
    std::vector<int> outputTensorIndex;
};

// layer 层抽象
class Layer {
public:
    explicit Layer(std::shared_ptr<LayerParams> param);

    Layer();

    ~Layer();

    // 此处的Tensor囊括了GPU的内存数据。

    // the init only run once
    virtual void init(const std::vector<Tensor>& input, std::vector<Tensor>& output);

    // and the forward can be run several times
    virtual void forward(const std::vector<Tensor>& input, std::vector<Tensor>& output);

    void setId(int id);
    int getId();
    std::string getName();
    LayerType getType();

protected:
    int layerId; // layer id 是layer在Net中前后顺序的序号，保存在Net的layerList中
    std::string layerName;
    LayerType layerType; // layerType是层类型
};

}

#endif //MINFER_LAYER_H
