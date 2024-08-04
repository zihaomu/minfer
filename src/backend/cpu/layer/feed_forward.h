//
// Created by mzh on 2024/7/23.
//

#ifndef MINFER_FEED_FORWARD_H
#define MINFER_FEED_FORWARD_H

#include "common_layer.h"

namespace minfer {

//
class FeedForwardLayer : public Layer
{
public:
    static std::shared_ptr<FeedForwardLayer> create(const std::shared_ptr<LayerParams> param);

    ~FeedForwardLayer();

    void init(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

    void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output) override;

private:
    FeedForwardLayer(const std::shared_ptr<LayerParams> param);
};

}

#endif //MINFER_FEED_FORWARD_H
