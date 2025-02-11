//
// Created by mzh on 2024/7/23.
//

#include <functional>
#include "feed_forward.h"

namespace minfer
{

FeedForwardLayer::FeedForwardLayer(const std::shared_ptr<FeedForwardLayerParams> param)
{
    embd_dim = param->embd_dim;
    ffn_dim = param->ffn_dim;
    rms_eps = param->rms_eps;

    param->norm.convertTo(norm, DT_32F);
    param->up.convertTo(up, DT_32F);
    param->gate.convertTo(gate, DT_32F);
    param->down.convertTo(down, DT_32F);

    activateType = param->actType;
}

void FeedForwardLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{

}

// TODO: add support start_pos
void FeedForwardLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output, int start_pos)
{
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    M_Assert(input[0]->type() == output[0]->type());

    MatShape in_shape = input[0]->shape();

    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[2] == embd_dim);
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");

    // pos_stripe 确定细节pos对计算对影响。
    // size_t pos_stripe = start_pos * total(in_shape, 1) * DT_ELEM_SIZE(input[0]->type());

    Mat x = *input[0];
    Mat x_norm = Mat(x.dims-1, x.size.p+1, DT_32F); // shape [bsz, seq_len, embed]
    float* p = (float *)x_norm.data;
    float* pi = (float *)(x.data);
    float * p_norm = (float *)norm.data;

    int seq_len = in_shape[1];

    // relu act
    std::function<float(const float )> act_func;
    if (activateType == ActivateType::SILU)
    {
        act_func = [&](const float v)
        {
            return std::max(0.f, v);
        };
    }
    else if (activateType == ActivateType::RELU)
    {
        act_func = [&](const float v)
        {
            return v/(1 + exp(-v));
        };
    }
    else
    {
        M_Error(NULL, "Un-supported activation type!");
    }

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

    // x1 = silu(self.linear1.forward(x))
    Mat x1 = gemm(x_norm, gate, false, true);

    float* p_x1 = (float *)x1.data;
    for (int i = 0; i < seq_len; i++)
    {
        p_x1[i] = act_func(p_x1[i]);
    }

    // x3 = self.linear3.forward(x)
    Mat x3 = gemm(x_norm, up, false, true);

    // x_out = self.linear2.forward(x1 * x3)
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims() - 1, out.size.p+1, out.type(), out.data);

    gemm(x1 * x3, down, false, true).copyTo(x_out);

//    out.print(10);
    Mat m = out + *input[0];
    m.copyTo(out);
//    out.print(10);
}

void FeedForwardLayer::finalize(const std::vector<Mat*>& input, std::vector<Mat*>& output)
{

}

FeedForwardLayer::~FeedForwardLayer()
{

}

std::shared_ptr<FeedForwardLayer> FeedForwardLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<FeedForwardLayerParams> ffn_param = std::dynamic_pointer_cast<FeedForwardLayerParams>(param);
    M_Assert(ffn_param && "FeedForwardLayerParams is empty!");
    M_Assert(ffn_param->type == LayerType::FFN);

    return std::shared_ptr<FeedForwardLayer>(new FeedForwardLayer(ffn_param));
}

}