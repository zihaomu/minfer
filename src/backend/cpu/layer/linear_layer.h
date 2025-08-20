//
// Created by mzh on 2024/1/24.
//

#ifndef MINFER_LINEAR_LAYER_H
#define MINFER_LINEAR_LAYER_H

#include "common_layer.h"

namespace minfer {

// Simpy layer to test create Net work.
class LinearLayer : public Layer
{
public:
    static std::shared_ptr<LinearLayer> create(const std::shared_ptr<LayerParams> param);

    ~LinearLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    // and the forward can be run several times
    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    int in_features;  // input, the number of input features
    int out_features; // output, the number of output features
    Mat w;           // weight matrix
    Mat b;           // bias vector
    bool transposeW = false; // 是否需要转置weight矩阵
    LinearLayer(const std::shared_ptr<LinearLayerParams> param);
};

}



#endif //MINFER_LINEAR_LAYER_H
