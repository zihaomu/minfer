//
// Created by mzh on 2024/1/24.
//

#ifndef MINFER_ADD_LAYER_H
#define MINFER_ADD_LAYER_H

#include "common_layer.h"

namespace minfer {

// Simpy layer to test create Net work.
class AddLayer : public Layer
{
public:
    static std::shared_ptr<AddLayer> create(const std::shared_ptr<LayerParams> param);

    ~AddLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    // and the forward can be run several times
    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    AddLayer(const std::shared_ptr<LayerParams> param);
    int inputNum = -1;
};

}



#endif //MINFER_ADD_LAYER_H
