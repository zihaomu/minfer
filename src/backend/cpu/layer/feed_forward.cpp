//
// Created by mzh on 2024/7/23.
//

#include "feed_forward.h"
#define FFN_DEBUG 0
namespace minfer
{

FeedForwardLayer::FeedForwardLayer(const std::shared_ptr<FeedForwardLayerParams> param)
{
    layerNamePrefix = "FeedForwardLayer_";
    embd_dim = param->embd_dim;
    ffn_dim = param->ffn_dim;
    rms_eps = param->rms_eps;

    param->norm.convertTo(norm, DT_32F);
    param->up.convertTo(up, DT_32F);
    param->gate.convertTo(gate, DT_32F);
    param->down.convertTo(down, DT_32F);

    activateType = param->actType;
#if ATTEN_DEBUG
    std::cout<<"print in init norm up gate down shape and params"<<std::endl;
    norm.print(2);
    up.print(2);
    gate.print(2);
    down.print(2);
#endif
}

void FeedForwardLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    // 设置同样的shape
    output[0]->setSize(*input[0]);
}

// TODO: add support start_pos
void FeedForwardLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
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
    Mat x_norm = rmsnorm(x, norm, rms_eps); // shape [1, seq_len, embed]

    // x1 = silu(self.linear1.forward(x))
    Mat x1 = gemm(x_norm, gate, false, true);

    if (activateType == ActivateType::RELU)
    {
        float* p_x1 = (float*)x1.data;
        const size_t total_elements = x1.total();
        for (size_t i = 0; i < total_elements; i++)
        {
            p_x1[i] = std::max(0.f, p_x1[i]);
        }
    }
    else if (activateType == ActivateType::SILU)
    {
        silu(x1, x1);
    }
    else
    {
        M_Error(NULL, "Un-supported activation type!");
    }

    // x3 = self.linear3.forward(x)
    Mat x3 = gemm(x_norm, up, false, true);

    // x_out = self.linear2.forward(x1 * x3)
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims(), out.size.p, out.type(), out.data);

    gemm(x1 * x3, down, false, true).copyTo(out);

    // std::cout<<"out gemm"<<std::endl;
    // out.print(10);

    // std::cout<<"out +="<<std::endl;
    out += *input[0];
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
