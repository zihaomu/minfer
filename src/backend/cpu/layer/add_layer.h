//
// Created by mzh on 2024/1/24.
//

#ifndef MINFER_ADD_LAYER_H
#define MINFER_ADD_LAYER_H

#include "minfer/layer.h"

namespace minfer {

// Simpy layer to test create Net work.
class AddLayer : public Layer
{
public:
    static std::shared_ptr<AddLayer> create(std::shared_ptr<LayerParams> param);

    ~AddLayer();

    void init(const std::vector<Tensor>& input, std::vector<Tensor>& output) override;

    // and the forward can be run several times
    void forward(const std::vector<Tensor>& input, std::vector<Tensor>& output) override;

private:
    AddLayer(std::shared_ptr<LayerParams> param);
    int inputNum = -1;
};



}



#endif //MINFER_ADD_LAYER_H
