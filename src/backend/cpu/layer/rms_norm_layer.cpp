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
    w = param->w;
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

    float* p = (float *)output[0]->data;
    float* pi = (float *)(input[0]->data);
    float * p_norm = (float *)w.data;

    int seq_len = in_shape[1];

    // rms-norm
    for (int i = 0; i < seq_len; i++)
    {
        float sum_f2 = 0;
        float* pi_s = pi + i * embd_dim;

        // extract np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        for (int j = 0; j < embd_dim; j++)
        {
            sum_f2 += pi_s[j] * pi_s[j];
        }

        float x1 = 1.f/sqrtf(sum_f2/embd_dim + rms_eps);

        for (int j = 0; j < embd_dim; j++)
        {
            p[i * embd_dim + j]= pi_s[j] * x1 * p_norm[j];
        }
    }
}

std::shared_ptr<RMSNormLayer> RMSNormLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<RMSNormLayerParams> r_param = std::dynamic_pointer_cast<RMSNormLayerParams>(param);

    M_Assert(r_param && "RMSNormLayerParams is empty!");
    M_Assert(r_param->type == LayerType::RMSNorm);

    return std::shared_ptr<RMSNormLayer>(new RMSNormLayer(r_param));

}

}