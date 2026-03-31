#ifndef MINFER_NET_H
#define MINFER_NET_H

#include "layer.h"
#include "context.h"
#include "mat.h"
#include "map"

namespace minfer
{

/// Net 类别
/* 例子代码：
 * Net nets = readNet("llama.gguf", RuntimePrecision::FP32);
 * Mat input = toknizer_input("I have a pen for", 2048);
 * nets.setInput(input);
 * nets.init();
 * Mat out;
 * while(true)
 * {
 *  nets.setInput(input);
 *  out = nets.forward();
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
    void createNet(const std::vector<std::shared_ptr<LayerParams> >& netParams, RuntimePrecision precision);

    /// 从模型文件中创建Net
    /// \param path
    /// \param precision runtime precision used at model load time, default fp32
    /// \param modelType
    /// \param kv_cache_cfg_path optional mobilekv cfg path, must be set before graph create.
    /// ⚠️目前只支持gguf一种模型格式
    void readNet(const std::string path,
                 RuntimePrecision precision = RuntimePrecision::FP32,
                 const std::string modelType = "gguf",
                 const std::string kv_cache_cfg_path = "");

    /// set input data with given mat index
    /// \param input
    /// \param mIndx defaule is -1, if the nets is single input.
    void setInput(const Mat input, const int mIndx = -1);

    void init();

    // 输入一个文本，输出token ids
    void encode(const std::string text, std::vector<int> &out_ids);

    void decode(const std::vector<int> &out_ids, std::string& out_text);

    void forward(Mat& out);

    // 生成模式
    void generate(Mat& out);

    // ====== Chat 生成接口 ======

    /// Prefill 阶段：处理完整 prompt，返回最后一个 token 的 logits
    Mat prefill(const std::vector<int>& token_ids);

    /// 自回归生成一步：输入上一步产出的 token，返回下一个 token 的 logits
    Mat step(int token_id);

    /// 重置所有 AttentionLayer 的 KV Cache，开始新一轮对话
    void resetKVCache();

    /// 设置 mobilekv cfg 路径（需在 readNet 前调用）
    void setKVCacheConfigPath(const std::string& cfg_path);

    RuntimePrecision getPrecision() const;

    Mat forward();

private:
    class NetImpl;
    NetImpl* impl; // 里面保存多种Backend，subnet，
};

//// 释放全局资源 TODO
//void releaseMinfer();

}

#endif //MINFER_NET_H
