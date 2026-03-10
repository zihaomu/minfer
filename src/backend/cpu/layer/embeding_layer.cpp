//
// Created by mzh on 2024/1/24.
//

#include "embeding_layer.h"

namespace minfer {

EmbeddingLayer::EmbeddingLayer(const std::shared_ptr<EmbeddingLayerParams> param)
{
    layerNamePrefix = "EmbeddingLayer_";
    M_Assert(param->type == LayerType::Embedding);
    getBasicInfo(param);

    vocab_dim = param->vocab_dim;
    embd_dim = param->embd_dim;

    MatShape w_shape = param->w.shape();
    M_Assert(w_shape.size() == 2);

    w.init(canonicalize_lookup_weight(param->w, vocab_dim, embd_dim),
           Int8QuantScheme::PerRow,
           false,
           param->precision);

    MatShape w_shape2 = w.active().shape();
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
    int inputNum = input.size();

    M_Assert(inputNum == 1);
    M_Assert(output.size() == 1);

    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 1 || in_shape.size() == 2); // [seq_len] or [batch, seq_len]

    int batch = in_shape.size() == 2 ? in_shape[0] : 1;
    int seq_len = in_shape.size() == 2 ? in_shape[1] : in_shape[0];

    // 设置同样的shape
    MatShape out_shape = {batch, seq_len, embd_dim};
    output[0]->setSize(out_shape);
}

void EmbeddingLayer::forward(const std::vector<Mat*> &input, std::vector<Mat*> &output)
{
    M_Assert(input.size() == 1);
    M_Assert(output.size() == 1);

    // 维度对齐
    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 1 || in_shape.size() == 2); // [seq_len] or [batch, seq_len]

    int batch = in_shape.size() == 2 ? in_shape[0] : 1;
    int seq_len = in_shape.size() == 2 ? in_shape[1] : in_shape[0];

    // TODO Multi batch
    M_Assert(batch == 1 && "Currently, only support single batch!");
    M_Assert(input[0]->type() == DT_32S); // 输入必须是整型
    M_Assert(output[0]->type() == DT_32F); // 输入必须是浮点型

    MatShape out_shape = output[0]->shape();

    M_Assert(out_shape.size() == 3); // [batch, seq_len, embd_dim]
    M_Assert(out_shape[0] == 1);
    M_Assert(out_shape[1] == seq_len); // seq_len should be same
    M_Assert(out_shape[2] == embd_dim);

    int* index = (int*)input[0]->data;
    float* output_ptr = (float*)output[0]->data;
    const RuntimePrecision precision = w.precision();
    const float* w_ptr_fp32 = precision == RuntimePrecision::FP32
        ? reinterpret_cast<const float*>(w.active().data)
        : nullptr;
    const hfloat* w_ptr_fp16 = precision == RuntimePrecision::FP16
        ? reinterpret_cast<const hfloat*>(w.active().data)
        : nullptr;
    const int8_t* w_ptr_int8 = precision == RuntimePrecision::INT8
        ? reinterpret_cast<const int8_t*>(w.active().data)
        : nullptr;
    const float* int8_scales = precision == RuntimePrecision::INT8
        ? reinterpret_cast<const float*>(w.int8Scales().data)
        : nullptr;

    for (int i = 0; i < seq_len; i++)
    {
        int word_id = index[i];
        float* embd = output_ptr + i * embd_dim;

        if (precision == RuntimePrecision::FP32)
        {
            memcpy(embd, w_ptr_fp32 + word_id * embd_dim, embd_dim * sizeof(float));
        }
        else if (precision == RuntimePrecision::FP16)
        {
            const hfloat* src = w_ptr_fp16 + word_id * embd_dim;
            for (int j = 0; j < embd_dim; ++j)
            {
                embd[j] = static_cast<float>(src[j]);
            }
        }
        else
        {
            const int8_t* src = w_ptr_int8 + word_id * embd_dim;
            const float scale = int8_scales[word_id];
            for (int j = 0; j < embd_dim; ++j)
            {
                embd[j] = static_cast<float>(src[j]) * scale;
            }
        }
        
        // Debug
        #if 0
        if (i < 2) {
            std::cout << "EmbLayer Token " << i << " (ID " << word_id << ") First 10: ";
            for (int k = 0; k < 10; k++) {
                std::cout << embd[k] << " ";
            }
            std::cout << std::endl;
        }
        #endif
    }
}

void EmbeddingLayer::setRuntimePrecision(RuntimePrecision precision)
{
    w.setPrecision(precision);
    Layer::setRuntimePrecision(precision);
}

std::shared_ptr<EmbeddingLayer> EmbeddingLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<EmbeddingLayerParams> e_param = std::dynamic_pointer_cast<EmbeddingLayerParams>(param);

    M_Assert(e_param && "EmbeddingLayerParams is empty!");
    M_Assert(e_param->type == LayerType::Embedding);

    return std::shared_ptr<EmbeddingLayer>(new EmbeddingLayer(e_param));
}

}
