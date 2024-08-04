//
// Created by moo on 2024/8/4.
//

#ifndef MINFER_ATTENTION_LAYER_H
#define MINFER_ATTENTION_LAYER_H

#include "common_layer.h"

namespace minfer {

/* MHA layer weights contain:
 * - wq
 * - wk
 * - wv
 * - woutput linear
 *
 * optional:
 * - bq
 * - bk
 * - bv
 * - boutput linear
 *
 * contains scalar:
 * - d_model
 * - num_head
 * - d_k = d_moedl / num_head
 * */
class AttentionLayer : public Layer
{
public:
    static std::shared_ptr<AttentionLayer> create(const std::shared_ptr<LayerParams> param);

    ~AttentionLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    int d_model;
    int num_head;
    int d_k; // d_k = d_model / num_head.
    AttentionLayer(const std::shared_ptr<AttentionLayerParams> param);
};

}

#endif //MINFER_ATTENTION_LAYER_H
