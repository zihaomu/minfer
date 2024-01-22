//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_LAYER_H
#define MINFER_LAYER_H

#include "string"

namespace minfer
{

// layer 层抽象
class Layer {
public:
    Layer();
    ~Layer();


    // 此处的Tensor囊括了GPU的内存数据。
    virtual void forward(const std::vector<Tensor>& input, std::vector<Tensor>& output) = 0;

private:
    int id;
    std::string name;


};

}

#endif //MINFER_LAYER_H
