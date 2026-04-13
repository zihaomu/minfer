//
// Created by Codex on 2026/4/9.
//

#ifndef MINFER_LM_HEAD_LAYER_H
#define MINFER_LM_HEAD_LAYER_H

#include "common_layer.h"
#include "runtime_weight.h"

namespace minfer {

class LmHeadLayer : public Layer
{
public:
    static std::shared_ptr<LmHeadLayer> create(const std::shared_ptr<LayerParams> param);

    ~LmHeadLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;
    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output, const InferenceContext& ctx) override;

    void setRuntimePrecision(RuntimePrecision precision) override;

private:
    void computeLogits(const Mat& x, Mat& out, const InferenceContext* ctx) const;
    void recordSubStage(const InferenceContext* ctx, const char* stage_name, double elapsed_us) const;
    void updateDecodeSelection(const Mat& logits, const InferenceContext& ctx) const;

    int in_features;
    int out_features;
    RuntimeWeight w;
    Mat b;

    explicit LmHeadLayer(const std::shared_ptr<LmHeadLayerParams> param);
};

}

#endif // MINFER_LM_HEAD_LAYER_H
