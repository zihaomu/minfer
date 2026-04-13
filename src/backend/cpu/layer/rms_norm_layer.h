//
// Created by mzh on 2024/1/24.
//

#ifndef MINFER_RMS_NORM_LAYER_H
#define MINFER_RMS_NORM_LAYER_H

#include "common_layer.h"
#include "runtime_weight.h"

namespace minfer {

// Simpy layer to test create Net work.
class RMSNormLayer : public Layer
{
public:
    static std::shared_ptr<RMSNormLayer> create(const std::shared_ptr<LayerParams> param);

    ~RMSNormLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    // and the forward can be run several times
    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void setRuntimePrecision(RuntimePrecision precision) override;

private:
    int embd_dim;
    float rms_eps;
    RuntimeWeight w;
    Mat align_input_scratch_;
    RMSNormLayer(const std::shared_ptr<RMSNormLayerParams> param);
};

}



#endif //MINFER_RMS_NORM_LAYER_H
