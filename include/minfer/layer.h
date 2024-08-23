//
// Created by mzh on 2024/1/22.
//

#ifndef MINFER_LAYER_H
#define MINFER_LAYER_H

#include <string>
#include <vector>
#include "mat.h"

namespace minfer
{

// TODO add all the layer type.
enum LayerType {
    UnSupported = -1,
    Input = 0, // input layer only has the output Mat
    Output = 1, // output layer only has the input Mat
    CONVOLUTION,
    Add,

    // llm layer
    Embedding,
    RMSNorm,
    RoPe,
    Softmax,
    FFN,
    Linear,
    Attention,
    LayerNorm,
};

enum ActivateType {
    UNKONW = -1,
    RELU = 0,
    SILU = 1,
    GELU = 2,
    SWISH = 3,

};

// 这里是否应当包含模型参数信息？
// 模型初始化中，应当保持
// 每一个层，如果有额外的需要，都应该重写这部分，从而添加额外的参数信息。
class LayerParams
{
public:
    virtual ~LayerParams() = default;

    LayerParams()
    :type(LayerType::UnSupported), inputIndex({}), outputIndex({})
    {}

    LayerParams(LayerType _type, std::vector<int> _inputIndex, std::vector<int> _outputIndex)
    :type(_type), inputIndex(_inputIndex), outputIndex(_outputIndex)
    {}

//    LayerParams(LayerType _type, std::vector<int> _inputIndex, std::vector<int> _outputIndex)
//    :type(_type), inputIndex(_inputIndex), outputIndex(_outputIndex)
//    {}

//    int layerId = -1;               // It will be set
    LayerType type;
    std::vector<int> inputIndex;
    std::vector<int> outputIndex;
    std::vector<Mat> weights;
};

class RMSNormLayerParams: public LayerParams
{
public:
    RMSNormLayerParams(std::vector<int> _inputIndex, std::vector<int> _outputIndex, int _embd_dim, float _rms_eps, Mat _w)
    :embd_dim(_embd_dim), rms_eps(_rms_eps), w(_w)
    {
        type = LayerType::RMSNorm;
        inputIndex = _inputIndex;
        outputIndex = _outputIndex;
    }

    int embd_dim; // output, embedding feature length.
    float rms_eps;// RMS norm layer
    Mat w;        // Embedding layer params
};

class EmbdLayerParams: public LayerParams
{
public:
    EmbdLayerParams(std::vector<int> _inputIndex, std::vector<int> _outputIndex, int _vocab_dim, int _embd_dim, Mat _w)
    :vocab_dim(_vocab_dim), embd_dim(_embd_dim), w(_w)
    {
        type = LayerType::Embedding;
        inputIndex = _inputIndex;
        outputIndex = _outputIndex;
    }

    int vocab_dim;  // input, the length of vocabulary
    int embd_dim;   // output, embedding feature length.
    Mat w;          // Embedding layer params
};

// TODO
class ConvLayerParams : public LayerParams
{
public:

};

// Multi-head Attention参数
class AttentionLayerParams : public LayerParams
{
public:
    AttentionLayerParams(std::vector<int> _inputIndex, std::vector<int> _outputIndex, int _max_seq_len, int _embd_dim,
                         int _head_count, int _head_count_kv, float _rms_eps,
                         Mat _norm, Mat _wq, Mat _wk, Mat _wv, Mat _wout, Mat _bq, Mat _bk, Mat _bv, Mat _bout)
    :max_seq_len(_max_seq_len), embd_dim(_embd_dim), head_count(_head_count), head_count_kv(_head_count_kv),
    rms_eps(_rms_eps), norm(_norm), wq(_wq), wk(_wk), wv(_wv), wout(_wout), bq(_bq), bk(_bk), bv(_bv), bout(_bout)
    {
        type = LayerType::Attention;
        inputIndex = _inputIndex;
        outputIndex = _outputIndex;
    }

    int max_seq_len;   // sequence max length
    int embd_dim;      // length of embedding feature
    int head_count;    // num_attention_heads
    int head_count_kv; // num_key_value_heads, the flag of Grouped Query Attention(GQA), if head_count_kv == head_count, the model will use Multi Head Attention(QHA), if head_count_kv==1, the model will use Multi Query Attention(MQA)
    float rms_eps;

    Mat norm;
    Mat wq;
    Mat wk;
    Mat wv;
    Mat wout;

    Mat bq;
    Mat bk;
    Mat bv;
    Mat bout;
};

class FeedForwardLayerParams : public LayerParams
{
public:
    FeedForwardLayerParams(std::vector<int> _inputIndex, std::vector<int> _outputIndex, ActivateType _actType,
                           int _embd_dim, int _ffn_dim, float _rms_eps, Mat _norm, Mat _gate, Mat _up, Mat _down)
    : actType(_actType), embd_dim(_embd_dim), ffn_dim(_ffn_dim), rms_eps(_rms_eps), norm(_norm), gate(_gate), up(_up), down(_down)
    {
        type = LayerType::FFN;
        inputIndex = _inputIndex;
        outputIndex = _outputIndex;
    }

    ActivateType actType;
    int embd_dim; // input embedding feature length
    int ffn_dim;  // ffn feature length
    float rms_eps;
    Mat norm;
    Mat gate;
    Mat up;
    Mat down;
};

// layer 层抽象
class Layer {
public:
    explicit Layer(const std::shared_ptr<LayerParams> param);

    Layer();

    ~Layer();

    // 此处的Tensor囊括了GPU的内存数据。

    // 初始化，根据输入Mat的shape，完成输出shape的计算，以及一些其他的初始化
    // TODO 加入这两项进入到推理中。
    // start pos 是llm模型的输入启始位置，而seqlen是此次推理seqlen的长度。
    virtual void init(const std::vector<Mat*>& input, std::vector<Mat*>& output);

    // 初始化完成之后，需要调用finalize函数完成一些初始化任务。
    virtual void finalize(const std::vector<Mat*>& input, std::vector<Mat*>& output);

    // and the forward can be run several times
    virtual void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output, int start_pos);

    // and the forward can be run several times
    virtual void forward(const std::vector<Mat*>& input, std::vector<Mat*>& output);

    void setId(int id);

    int getId();

    std::string getName();

    LayerType getType();

protected:
    void getBasicInfo(const std::shared_ptr<LayerParams> param);
    int layerId; // layer id 是layer在Net中前后顺序的序号，保存在Net的layerList中
    std::string layerName;
    LayerType layerType; // layerType是层类型
};

}

#endif //MINFER_LAYER_H
