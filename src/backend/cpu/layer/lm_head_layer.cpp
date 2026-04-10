//
// Created by Codex on 2026/4/9.
//

#include "lm_head_layer.h"

#include <algorithm>

namespace minfer {

LmHeadLayer::LmHeadLayer(const std::shared_ptr<LmHeadLayerParams> param)
{
    layerNamePrefix = "LmHeadLayer_";
    M_Assert(param->type == LayerType::LmHead);
    getBasicInfo(param);

    in_features = param->in_features;
    out_features = param->out_features;

    w.init(canonicalize_linear_weight(param->w, out_features, in_features),
           Int8QuantScheme::PerRow,
           true,
           param->precision);

    if (!param->b.empty())
    {
        M_Assert(param->b.shape().size() == 1 && param->b.shape()[0] == out_features);
        param->b.convertTo(b, DT_32F);
    }
}

LmHeadLayer::~LmHeadLayer()
{
}

void LmHeadLayer::init(const std::vector<Mat*>& input, std::vector<Mat*>& output)
{
    M_Assert(input.size() == output.size() && input.size() == 1);

    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() >= 2);

    MatShape output_shape = in_shape;
    output_shape.back() = out_features;

    output[0]->setSize(output_shape);
}

void LmHeadLayer::forward(const std::vector<Mat*>& input, std::vector<Mat*>& output)
{
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    M_Assert(input[0]->type() == output[0]->type());

    Mat x = *input[0];
    Mat out = *output[0];

    MatShape in_shape = x.shape();
    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");
    M_Assert(in_shape[2] == in_features);

    w.gemmNT(x).copyTo(out);

    if (!b.empty())
    {
        out = out + b;
    }
}

void LmHeadLayer::forward(const std::vector<Mat*>& input, std::vector<Mat*>& output, const InferenceContext& ctx)
{
    if (ctx.phase == InferPhase::Decode &&
        ctx.decode_output_mode != DecodeOutputMode::FullLogits &&
        ctx.decode_selection != nullptr &&
        w.selectNT(*input[0],
                   ctx.decode_output_mode,
                   ctx.top_k,
                   b.empty() ? nullptr : &b,
                   *ctx.decode_selection))
    {
        return;
    }

    forward(input, output);
    updateDecodeSelection(*output[0], ctx);
}

void LmHeadLayer::updateDecodeSelection(const Mat& logits, const InferenceContext& ctx) const
{
    if (ctx.decode_output_mode == DecodeOutputMode::FullLogits || ctx.decode_selection == nullptr)
    {
        return;
    }

    M_Assert(logits.dims == 3);
    M_Assert(logits.size[0] == 1);
    M_Assert(logits.type() == DT_32F);

    DecodeSelection& selection = *ctx.decode_selection;
    selection.reset(ctx.decode_output_mode, ctx.top_k);

    const int seq_len = logits.size[1];
    const int vocab_size = logits.size[2];
    M_Assert(seq_len > 0);
    M_Assert(vocab_size > 0);

    const float* row = reinterpret_cast<const float*>(logits.data) +
                       static_cast<size_t>(seq_len - 1) * static_cast<size_t>(vocab_size);

    if (ctx.decode_output_mode == DecodeOutputMode::ArgMax)
    {
        for (int v = 0; v < vocab_size; ++v)
        {
            const float logit = row[v];
            if (logit > selection.logit || (logit == selection.logit && (selection.token_id < 0 || v < selection.token_id)))
            {
                selection.token_id = v;
                selection.logit = logit;
            }
        }
        selection.ready = true;
        return;
    }

    const int k = std::max(1, std::min(ctx.top_k, vocab_size));
    auto& topk = selection.top_k;
    topk.clear();
    topk.reserve(k);

    auto better = [](const DecodeCandidate& lhs, const DecodeCandidate& rhs) {
        if (lhs.logit != rhs.logit)
        {
            return lhs.logit > rhs.logit;
        }
        return lhs.token_id < rhs.token_id;
    };

    for (int v = 0; v < vocab_size; ++v)
    {
        DecodeCandidate candidate = {v, row[v]};

        auto it = std::find_if(topk.begin(), topk.end(),
                               [&](const DecodeCandidate& existing) { return better(candidate, existing); });
        if (it != topk.end())
        {
            topk.insert(it, candidate);
        }
        else if (static_cast<int>(topk.size()) < k)
        {
            topk.push_back(candidate);
        }

        if (static_cast<int>(topk.size()) > k)
        {
            topk.pop_back();
        }
    }

    M_Assert(!topk.empty());
    selection.token_id = topk.front().token_id;
    selection.logit = topk.front().logit;
    selection.ready = true;
}

void LmHeadLayer::setRuntimePrecision(RuntimePrecision precision)
{
    w.setPrecision(precision);
    Layer::setRuntimePrecision(precision);
}

std::shared_ptr<LmHeadLayer> LmHeadLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<LmHeadLayerParams> lm_param = std::dynamic_pointer_cast<LmHeadLayerParams>(param);

    M_Assert(lm_param && "LmHeadLayerParams is empty!");
    M_Assert(lm_param->type == LayerType::LmHead);

    return std::shared_ptr<LmHeadLayer>(new LmHeadLayer(lm_param));
}

}
