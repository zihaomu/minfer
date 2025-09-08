//
// Created by moo on 2024/8/4.
//

#ifndef MINFER_ATTENTION_LAYER_H
#define MINFER_ATTENTION_LAYER_H

#include "minfer.h"
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

    void finalize(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    Mat norm;
    Mat wq;
    Mat wk;
    Mat wv;
    Mat wout;

    bool has_bias;
    Mat bq;
    Mat bk;
    Mat bv;
    Mat bout;

    int max_seq_len;       // length of sequence.
    int embd_dim;      // length of embedding feature
    int head_count;    // num_attention_heads
    int head_count_kv; // num_key_value_heads, the flag of Grouped Query Attention(GQA), if head_count_kv == head_count, the model will use Multi Head Attention(QHA), if head_count_kv==1, the model will use Multi Query Attention(MQA)
    float rms_eps;     // norm eps value
    int repeat_kv;     // for Group query attention, need to broadcasting kv tensor.
    int embd_dim_head;     // embd_dim of each head. d_k otherwise.
    int embd_dim_kv;       // embd_dim of kv

    int start_pos = 0;     // 标志从哪里开始开始推理
    AttentionLayer(const std::shared_ptr<AttentionLayerParams> param);
};

}

#endif //MINFER_ATTENTION_LAYER_H
