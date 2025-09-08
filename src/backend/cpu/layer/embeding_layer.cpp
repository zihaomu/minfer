//
// Created by mzh on 2024/1/24.
//

#include "embeding_layer.h"

namespace minfer {

EmbeddingLayer::EmbeddingLayer(const std::shared_ptr<EmbeddingLayerParams> param)
{
    M_Assert(param->type == LayerType::Embedding);
    getBasicInfo(param);

    vocab_dim = param->vocab_dim;
    embd_dim = param->embd_dim;

    MatShape w_shape = param->w.shape();
    M_Assert(w_shape.size() == 2);


    auto tt = param->w.total();
    auto t = param->w.type();

    Mat wFp32;
    param->w.convertTo(wFp32, DT_32F);

    // 有的模型会将embedding的weight设置为[embd_dim, vocab_dim]，有的模型会设置为[vocab_dim, embd_dim]
    if (w_shape[0] == vocab_dim && w_shape[1] == embd_dim)
    {
        // 这种情况是[vocab_dim, embd_dim]
        w = wFp32;
    }
    else if (w_shape[0] == embd_dim && w_shape[1] == vocab_dim)
    {
        // 这种情况是[embd_dim, vocab_dim]
        std::vector<int> new_shape = {vocab_dim, embd_dim};
        w = transpose(wFp32);
    }
    else
        M_Error(NULL, "EmbeddingLayer weight shape is not supported! ");

    MatShape w_shape2 = w.shape();
    M_Assert(w_shape2[0] == vocab_dim);
    M_Assert(w_shape2[1] == embd_dim);
}

EmbeddingLayer::~EmbeddingLayer()
{

}
/* ebeding shape
 * 输入：token ids
 * 输出：[B, L, H]
 *
 * 其中，prefill阶段，token ids为多个，= [B, L, H]
 * decode 阶段 token id为1个，输出shape = [B, 1, H]
 *
 */
void EmbeddingLayer::init(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    // pre check
    inputNum = input.size();

    M_Assert(inputNum == 1);
    M_Assert(output.size() == 1);

    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 2); // [batch, seq_len]

    // 设置同样的shape
    MatShape out_shape = {in_shape[0], in_shape[1], embd_dim};
    output[0]->setSize(out_shape);
}

void EmbeddingLayer::forward(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    M_Assert(input.size() == 1);
    M_Assert(output.size() == 1);

    // 维度对齐
    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 2); // [batch, seq_len]

    // TODO Multi batch
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");
    M_Assert(input[0]->type() == DT_32S); // 输入必须是整型
    M_Assert(output[0]->type() == DT_32F); // 输入必须是整型

    MatShape out_shape = output[0]->shape();

    M_Assert(out_shape.size() == 3); // [batch, seq_len, embd_dim]
    M_Assert(out_shape[0] == 1);
    M_Assert(out_shape[1] == in_shape[1]); // seq_len should be same
    M_Assert(out_shape[2] == embd_dim);

    size_t seq_len = in_shape[1];

    int* index = (int*)input[0]->data;
    float* w_ptr = (float*)w.data;
    float* output_ptr = (float*)output[0]->data;

    for (int i = 0; i < seq_len; i++)
    {
        int word_id = index[i];
        float* embd = output_ptr + i * embd_dim;

        memcpy(embd, w_ptr + word_id * embd_dim, embd_dim * sizeof(float));
    }
}

std::shared_ptr<EmbeddingLayer> EmbeddingLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<EmbeddingLayerParams> e_param = std::dynamic_pointer_cast<EmbeddingLayerParams>(param);

    M_Assert(e_param && "EmbeddingLayerParams is empty!");
    M_Assert(e_param->type == LayerType::Embedding);

    return std::shared_ptr<EmbeddingLayer>(new EmbeddingLayer(e_param));
}

}