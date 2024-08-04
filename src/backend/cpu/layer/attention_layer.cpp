//
// Created by moo on 2024/8/4.
//

#include "attention_layer.h"

namespace minfer {

AttentionLayer::AttentionLayer(const std::shared_ptr<AttentionLayerParams> param)
{

}

void AttentionLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{

}

void AttentionLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{

}

AttentionLayer::~AttentionLayer()
{

}

std::shared_ptr<AttentionLayer> AttentionLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<AttentionLayerParams> attn_param = std::dynamic_pointer_cast<AttentionLayerParams>(param);

    return std::shared_ptr<AttentionLayer>(new AttentionLayer(attn_param));
}

}

