//
// Created by mzh on 2024/3/28.
//

#ifndef MINFER_INPUT_LAYER_H
#define MINFER_INPUT_LAYER_H

#include "common_layer.h"

namespace minfer {

// InputLayer only has output layer, do not have the input layer.
class InputLayer : public Layer
{
public:
    static std::shared_ptr<InputLayer> create(const std::shared_ptr<LayerParams> param);

    ~InputLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    InputLayer(const std::shared_ptr<LayerParams> param);
};

}

#endif //MINFER_INPUT_LAYER_H
