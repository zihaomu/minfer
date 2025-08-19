//
// Created by mzh on 2024/1/24.
//

#ifndef MINFER_EMBEDDING_LAYER_H
#define MINFER_EMBEDDING_LAYER_H

#include "minfer.h"
#include "common_layer.h"

namespace minfer {

// Simpy layer to test create Net work.
class EmbeddingLayer : public Layer
{
public:
    static std::shared_ptr<EmbeddingLayer> create(const std::shared_ptr<LayerParams> param);

    ~EmbeddingLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    // and the forward can be run several times
    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    int vocab_dim;  // input, the length of vocabulary
    int embd_dim;   // output, embedding feature length.
    Mat w;          // Embedding layer params

    EmbeddingLayer(const std::shared_ptr<EmbeddingLayerParams> param);
    int inputNum = -1;
};

}



#endif //MINFER_EMBEDDING_LAYER_H
