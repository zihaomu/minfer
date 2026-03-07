//
// Created by mzh on 2024/1/24.
//

#include "rms_norm_layer.h"

namespace minfer {

RMSNormLayer::RMSNormLayer(const std::shared_ptr<RMSNormLayerParams> param)
{
    layerNamePrefix = "RMSNormLayer_";
    M_Assert(param->type == LayerType::RMSNorm);
    getBasicInfo(param);

    embd_dim = param->embd_dim;
    rms_eps = param->rms_eps;
    MatShape w_shape = param->w.shape();
    M_Assert(w_shape.size() == 1 && w_shape[0] == embd_dim);
    w.init(param->w, Int8QuantScheme::PerTensor);
}

RMSNormLayer::~RMSNormLayer()
{

}

void RMSNormLayer::init(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    // pre check
    int input_num = input.size();

    M_Assert(input_num == 1);
    M_Assert(output.size() == 1);

    // 设置同样的shape
    output[0]->setSize(*input[0]);
}

void RMSNormLayer::forward(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    // TODO finish the following code!

    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    M_Assert(input[0]->type() == output[0]->type());

    MatShape in_shape = input[0]->shape();

    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[2] == embd_dim);
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");

    if (w.usesInt8())
    {
        rmsnorm(*input[0], w.active(), w.int8Scales(), *output[0], rms_eps);
    }
    else
    {
        rmsnorm(*input[0], w.active(), *output[0], rms_eps);
    }
}

std::shared_ptr<RMSNormLayer> RMSNormLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<RMSNormLayerParams> r_param = std::dynamic_pointer_cast<RMSNormLayerParams>(param);

    M_Assert(r_param && "RMSNormLayerParams is empty!");
    M_Assert(r_param->type == LayerType::RMSNorm);

    return std::shared_ptr<RMSNormLayer>(new RMSNormLayer(r_param));

}

void RMSNormLayer::setRuntimePrecision(RuntimePrecision precision)
{
    Layer::setRuntimePrecision(precision);
    w.setPrecision(precision);
}

}
