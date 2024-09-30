#ifndef MINFER_NET_H
#define MINFER_NET_H

#include "layer.h"
#include "mat.h"
#include "map"

namespace minfer
{

/// Net 类别
/* 例子代码：
 * Net net = readNet("llama.gguf");
 * Mat input = toknizer_input("I have a pen for", 2048);
 * net.setInput(input);
 * net.init();
 * Mat out;
 * while(true)
 * {
 *  net.setInput(input);
 *  out = net.forward();
 *
 *  bool ifStop = false;
 *  std::string chat_out = tokizer_output(out, ifStop);
 *  std::cout<<"say: "<<chat_out<<std::endl;
 *  if (ifStop)
 *      break;
 *  input = out;
 * }
 * */
class Net {
public:
    Net();
    ~Net();

    // create new layer, and return layerId
    int createLayer(std::shared_ptr<LayerParams> param);

    void createNet(const std::vector<std::shared_ptr<LayerParams> >& netParams);

    /// 从模型文件中创建Net
    /// \param path
    /// \param modelType
    /// ⚠️目前只支持gguf一种模型格式
    void readNet(const std::string path, const std::string modelType = "gguf");

    /// set input data with given mat index
    /// \param input
    /// \param mIndx defaule is -1, if the net is single input.
    void setInput(const Mat input, const int mIndx = -1);

    void init();

    void forward(Mat& out);

    // 生成模式
    void generate(Mat& out);

    Mat forward();

private:
    class NetImpl;
    NetImpl* impl; // 里面保存多种Backend，subnet，
};

// 释放全局资源
void releaseMinfer();

}

#endif //MINFER_NET_H
