//
// Created by moo on 2024/8/4.
//

#include "attention_layer.h"
namespace minfer {

AttentionLayer::AttentionLayer(const std::shared_ptr<AttentionLayerParams> param)
{
    embd_dim = param->embd_dim;
    head_count = param->head_count;
    head_count_kv = param->head_count_kv;
    rms_eps = param->rms_eps;

    param->norm.convertTo(norm, DT_32F);

    param->wq.convertTo(wq, DT_32F);
    param->wk.convertTo(wk, DT_32F);
    param->wv.convertTo(wv, DT_32F);
    param->wout.convertTo(wout, DT_32F);

    param->bq.convertTo(bq, DT_32F);
    param->bk.convertTo(bk, DT_32F);
    param->bv.convertTo(bv, DT_32F);
    param->bout.convertTo(bout, DT_32F);
}

/* forward function contains two operator, RMSnorm and attention.
 *
 * */
void AttentionLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    // shape check
    M_Assert(input.size() == 1 && input[0]);
    MatShape in_shape = input[0]->shape();

    M_Assert(in_shape.size() == 2);
    M_Assert(in_shape[1] == embd_dim);

    // TODO support multi-type Mat. Current only fp16 is supported.
    M_Assert(input[0]->type() == DT_16F);

    // implementation the rms norm
    Mat x2 = *input[0] * *input[0];
    Mat x_norm = Mat(input[0]->dims, input[0]->size.p, DT_32F);

    float* p = (float *)x_norm.data;
    uint16_t* pi = (uint16_t*)input[0]->data;
    float * p_norm = (float *)norm.data;

    int iw = in_shape[1];
    for (int i = 0; i < in_shape[0]; i++)
    {
        float sum_f2 = 0;
        // extract np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        for (int j = 0; j < iw; j++)
        {
            float f = fp16_to_fp32(pi[i * iw + j]);
            sum_f2 += f * f;
        }

        float x1 = 1.f/sqrt(sum_f2/iw + rms_eps);

        for (int j = 0; j < iw; j++)
        {
            float f = fp16_to_fp32(pi[i * iw + j]);
            p[i * iw + j]= f * x1 * p_norm[j];
        }
    }

    // implementation Q K V linear


    // implementation Q K RoPe

    // implementation Q K matmul and mask

    // implementation softmax

    // implementation matmul V

    // implementation out linear.
}

void precompute_freq_cis(int dim, int end, int rms_eps)
{

}

void AttentionLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    //
}

AttentionLayer::~AttentionLayer()
{

}

std::shared_ptr<AttentionLayer> AttentionLayer::create(const std::shared_ptr<LayerParams> param)
{
    std::shared_ptr<AttentionLayerParams> attn_param = std::dynamic_pointer_cast<AttentionLayerParams>(param);
    M_Assert(attn_param && "AttentionLayerParams is empty!");
    M_Assert(attn_param->type == LayerType::Attention);

    return std::shared_ptr<AttentionLayer>(new AttentionLayer(attn_param));
}

}

