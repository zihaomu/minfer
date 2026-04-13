//
// Created by mzh on 2024/7/23.
//

#include "feed_forward.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>

#define FFN_DEBUG 0
namespace minfer
{

namespace {

bool ffn_decode_pair_enabled()
{
    const char* env = std::getenv("MINFER_FFN_DECODE_PAIR");
    if (env == nullptr || env[0] == '\0')
    {
        return false;
    }
    return !(env[0] == '0' && env[1] == '\0');
}

void apply_ffn_activation_product(ActivateType activate_type,
                                  const Mat& gate_input,
                                  const Mat& up_input,
                                  Mat& fused_output)
{
    M_Assert(gate_input.type() == DT_32F);
    M_Assert(up_input.type() == DT_32F);
    M_Assert(fused_output.type() == DT_32F);
    M_Assert(gate_input.shape() == up_input.shape());
    M_Assert(gate_input.shape() == fused_output.shape());

    const float* gate_ptr = reinterpret_cast<const float*>(gate_input.data);
    const float* up_ptr = reinterpret_cast<const float*>(up_input.data);
    float* fused_ptr = reinterpret_cast<float*>(fused_output.data);

    const size_t total_elements = fused_output.total();
    if (activate_type == ActivateType::RELU)
    {
        for (size_t i = 0; i < total_elements; ++i)
        {
            fused_ptr[i] = std::max(0.0f, gate_ptr[i]);
        }
    }
    else if (activate_type == ActivateType::SILU)
    {
        silu(gate_input, fused_output);
    }
    else
    {
        M_Error(NULL, "Un-supported activation type!");
    }

    multiply(fused_output, up_input, fused_output);
}

}  // namespace

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

    Mat gate_proj = gate.gemmNT(x_norm);
    Mat gate_activated = gate_proj;
    if (runtimePrecision != RuntimePrecision::FP32)
    {
        align_precision_sensitive_input(gate_proj, runtimePrecision, gate_align_scratch_);
        gate_activated = gate_align_scratch_;
    }
    Mat up_proj = up.gemmNT(x_norm);
    apply_ffn_activation_product(activateType, gate_activated, up_proj, gate_activated);

    Mat out = *output[0];
    down.gemmNT(gate_activated, out);

    // std::cout<<"out gemm"<<std::endl;
    // out.print(10);

    // std::cout<<"out +="<<std::endl;
    out += *input[0];
}

void FeedForwardLayer::forward(const std::vector<Mat*>& input,
                               std::vector<Mat*>& output,
                               const InferenceContext& ctx)
{
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    if (tryDecodeFusedForward(*input[0], *output[0], ctx))
    {
        return;
    }

    forward(input, output);
}

bool FeedForwardLayer::tryDecodeFusedForward(const Mat& x, Mat& out, const InferenceContext& ctx)
{
    if (!ffn_decode_pair_enabled())
    {
        return false;
    }

    if (ctx.phase != InferPhase::Decode || ctx.seq_len != 1)
    {
        return false;
    }

    if (x.type() != DT_32F || out.type() != DT_32F)
    {
        return false;
    }

    const MatShape in_shape = x.shape();
    if (in_shape.size() != 3 || in_shape[0] != 1 || in_shape[1] != 1 || in_shape[2] != embd_dim)
    {
        return false;
    }

    Mat x_norm = norm.usesInt8()
        ? rmsnorm(x, norm.active(), norm.int8Scales(), rms_eps)
        : rmsnorm(x, norm.active(), rms_eps);

    Mat gate_raw;
    Mat up_raw;
    if (!gate.gemmNTPair(x_norm, up, gate_raw, up_raw))
    {
        return false;
    }

    Mat gate_input = gate_raw;
    if (runtimePrecision != RuntimePrecision::FP32)
    {
        align_precision_sensitive_input(gate_raw, runtimePrecision, gate_align_scratch_);
        gate_input = gate_align_scratch_;
    }
    apply_ffn_activation_product(activateType, gate_input, up_raw, gate_raw);

    down.gemmNT(gate_raw, out);
    out += x;
    return true;
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
