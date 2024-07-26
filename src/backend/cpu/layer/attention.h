//
// Created by mzh on 2024/7/23.
//

#ifndef MINFER_ATTENTION_H
#define MINFER_ATTENTION_H

#include "common_layer.h"

namespace minfer {

//
class Attention : public Layer
{
public:
    static std::shared_ptr<Attention> create(const std::shared_ptr<LayerParams> param);

    ~Attention();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    Attention(const std::shared_ptr<LayerParams> param);
};

/* MHA layer weights contain:
 * - q
 * - k
 * - v
 * - output linear
 *
 * contains scalar:
 * - d_model
 * - num_head
 * - d_k = d_moedl / num_head
 * */
class MultiHeadAttention : public Layer
{
public:
    static std::shared_ptr<MultiHeadAttention> create(const std::shared_ptr<LayerParams> param);

    ~MultiHeadAttention();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    int d_model;
    int num_head;
    int d_k; // d_k = d_model / num_head.
private:
    MultiHeadAttention(const std::shared_ptr<LayerParams> param);
};
}

#endif //MINFER_ATTENTION_H
