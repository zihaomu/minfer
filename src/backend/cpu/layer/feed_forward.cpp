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
    getBasicInfo(param);
    embd_dim = param->embd_dim;
    ffn_dim = param->ffn_dim;
    rms_eps = param->rms_eps;

    norm.init(param->norm, Int8QuantScheme::PerTensor, false, param->precision);
    up.init(canonicalize_linear_weight(param->up, ffn_dim, embd_dim), Int8QuantScheme::PerRow, true, param->precision);
    gate.init(canonicalize_linear_weight(param->gate, ffn_dim, embd_dim), Int8QuantScheme::PerRow, true, param->precision);
    down.init(canonicalize_linear_weight(param->down, embd_dim, ffn_dim), Int8QuantScheme::PerRow, true, param->precision);

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
    Mat x_norm = norm.usesInt8()
        ? rmsnorm(x, norm.active(), norm.int8Scales(), rms_eps)
        : rmsnorm(x, norm.active(), rms_eps); // shape [1, seq_len, embed]

    // x1 = silu(self.linear1.forward(x))
    Mat x1 = gate.gemmNT(x_norm);
    Mat x1_aligned = runtimePrecision == RuntimePrecision::FP32
        ? x1
        : align_precision_sensitive_input(x1, runtimePrecision);

    if (activateType == ActivateType::RELU)
    {
        x1 = x1_aligned;
        float* p_x1 = (float*)x1.data;
        const size_t total_elements = x1.total();
        for (size_t i = 0; i < total_elements; i++)
        {
            p_x1[i] = std::max(0.f, p_x1[i]);
        }
    }
    else if (activateType == ActivateType::SILU)
    {
        silu(x1_aligned, x1);
    }
    else
    {
        M_Error(NULL, "Un-supported activation type!");
    }

    // x3 = self.linear3.forward(x)
    Mat x3 = up.gemmNT(x_norm);

    // x_out = self.linear2.forward(x1 * x3)
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims(), out.size.p, out.type(), out.data);

    down.gemmNT(x1 * x3).copyTo(out);

    // std::cout<<"out gemm"<<std::endl;
    // out.print(10);

    // std::cout<<"out +="<<std::endl;
    out += *input[0];
}

void FeedForwardLayer::finalize(const std::vector<Mat*>& input, std::vector<Mat*>& output)
{

}

void FeedForwardLayer::setRuntimePrecision(RuntimePrecision precision)
{
    norm.setPrecision(precision);
    gate.setPrecision(precision);
    up.setPrecision(precision);
    down.setPrecision(precision);
    Layer::setRuntimePrecision(precision);
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
