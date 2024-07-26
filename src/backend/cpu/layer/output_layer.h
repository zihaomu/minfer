//
// Created by mzh on 2024/3/28.
//

#ifndef MINFER_OUTPUT_LAYER_H
#define MINFER_OUTPUT_LAYER_H

#include "common_layer.h"

namespace minfer {

// OutputLayer only has input layer, do not have the output layer.
class OutputLayer : public Layer
{
public:
    static std::shared_ptr<OutputLayer> create(const std::shared_ptr<LayerParams> param);

    ~OutputLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    OutputLayer(const std::shared_ptr<LayerParams> param);
};

}

#endif //MINFER_OUTPUT_LAYER_H
