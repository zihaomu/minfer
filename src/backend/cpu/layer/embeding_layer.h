//
// Created by mzh on 2024/1/24.
//

#ifndef MINFER_EMBEDDING_LAYER_H
#define MINFER_EMBEDDING_LAYER_H

#include "minfer.h"
#include "common_layer.h"

namespace minfer {

/* Embeding 层分别在 输入和输出中负责 loopup和projection 角色
 * 其中，在输入时，需要将token ids转换为 hidden，这时候是lookup
 * 在输出时，需要将hidden转换为 token ids，这时候时projection
 */
class EmbeddingLayer : public Layer
{
    enum Mode {
        LOOKUP = 0,
        PROJECTION,
    };
public:
    static std::shared_ptr<EmbeddingLayer> create(const std::shared_ptr<LayerParams> param);

    ~EmbeddingLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    // and the forward can be run several times
    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    EmbeddingLayer(const std::shared_ptr<EmbeddingLayerParams> param);

    int vocab_dim;  // input, the length of vocabulary
    int embd_dim;   // output, embedding feature length.
    Mat w;          // Embedding layer params

    Mode model = LOOKUP; // 目前仅支持 loopup模式，projection 情况目前是用 linear层实现。
};

}



#endif //MINFER_EMBEDDING_LAYER_H
