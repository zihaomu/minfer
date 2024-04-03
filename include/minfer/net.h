#ifndef MINFER_NET_H
#define MINFER_NET_H

#include "layer.h"
#include "mat.h"

namespace minfer
{

// Net抽象
class Net {
public:
    Net();
    ~Net();

    // create new layer, and return layerId
    int createLayer(std::shared_ptr<LayerParams> param);

    void readNet(const std::string path, const std::string modelType);

//    void setInput(const Mat input, const std::string name = {});

    /// set input data with given mat index
    /// \param input
    /// \param mIndx defaule is -1, if the net is single input, default value is en
    void setInput(const Mat input, const int mIndx = -1);

    void init();

    void forward(Mat& out);

    Mat forward();

private:
    class NetImpl;
    NetImpl* impl; // 里面保存多种Backend，subnet，
};

// 释放全局资源
void releaseMinfer();

}

#endif //MINFER_NET_H
