//
// Created by moo on 2024/8/4.
//

#include "attention_layer.h"
#include "autobuffer.h"

#define ATTEN_DEBUG 1
namespace minfer {

AttentionLayer::AttentionLayer(const std::shared_ptr<AttentionLayerParams> param)
{
    max_seq_len = param->max_seq_len;
    embd_dim = param->embd_dim;
    head_count = param->head_count;
    M_Assert(embd_dim % head_count == 0);

    head_count_kv = param->head_count_kv;
    rms_eps = param->rms_eps;

    M_Assert(head_count % head_count_kv == 0);
    repeat_kv = head_count / head_count_kv;
    embd_dim_head = embd_dim / head_count;
    embd_dim_kv = embd_dim_head * head_count_kv;

    param->norm.convertTo(norm, DT_32F);

    param->wq.convertTo(wq, DT_32F);
    param->wk.convertTo(wk, DT_32F);
    param->wv.convertTo(wv, DT_32F);
    param->wout.convertTo(wout, DT_32F);

    param->bq.convertTo(bq, DT_32F);
    param->bk.convertTo(bk, DT_32F);
    param->bv.convertTo(bv, DT_32F);
    param->bout.convertTo(bout, DT_32F);

#if ATTEN_DEBUG
    std::cout<<"print in init q k v out shape and params"<<std::endl;
    wq.print(2);
    wk.print(2);
    wv.print(2);
    wout.print(2);
#endif
}

void AttentionLayer::finalize(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{

}

Mat softmax(Mat inp)
{
    Mat out = inp.clone();
    M_Assert(inp.type() == DT_32F);
    size_t total = inp.total();
    const float* p_i = (const float*)inp.data;
    float* p_o = (float*)out.data;

    float max_val = p_i[0];
    for (int i = 1; i < total; i++)
    {
        if (p_i[i] > max_val)
            max_val = p_i[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < total; i++)
    {
        p_o[i] = expf(p_i[i] - max_val);
        sum += p_o[i];
    }
    // normalize
    for (int i = 0; i < total; i++) {
        p_o[i] /= sum;
    }

    return out;
}

/* forward function contains two operator, RMSnorm and attention.
 * forward contain start_pos and sequence len, how to set the sequence len to the forward?
 * */
// TODO take into account the kv_head is different with head_count.
// TODO try to use bias params
void AttentionLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output, int start_pos)
{
    // shape check
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    MatShape in_shape = input[0]->shape();

    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[2] == embd_dim);
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");

    // TODO support multi-type Mat. Current only fp16 is supported.
    M_Assert(input[0]->type() == DT_32F);

    // step0: implementation the rms norm
    // xq shape is [bsz, seq, embed]
    Mat x = *input[0];
    Mat x_norm = Mat(x.dims - 1, x.size.p + 1, DT_32F); // shape [bsz, seq_len, embed]

    float* p = (float *)x_norm.data;
    float* pi = (float *)x.data;
    float * p_norm = (float *)norm.data;

    int seq_len = in_shape[1];

    for (int i = 0; i < seq_len; i++)
    {
        float sum_f2 = 0;
        float* pi_s = pi + i * embd_dim;

        // extract np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)
        for (int j = 0; j < embd_dim; j++)
        {
            sum_f2 += pi_s[j] * pi_s[j];
        }

        float x1 = 1.f/sqrtf(sum_f2/embd_dim + rms_eps);

        for (int j = 0; j < embd_dim; j++)
        {
            p[i * embd_dim + j]= pi_s[j] * x1 * p_norm[j];
        }
    }

    // implementation Q K V linear

    Mat x_q = gemm(x_norm, wq); // xq shape is [bsz, seq, embed], x_norm shape is [embed, embed], after shape, is the same.
    Mat x_k = gemm(x_norm, wk); // k and v may has different shape with q, use Group-query attention.
    Mat x_v = gemm(x_norm, wv); // wk and wv shape is [embed, embd_dim_kv], x_k = [bsz, seq, embd_dim_kv]

#if ATTEN_DEBUG
    std::cout<<"print in init xq, xk xv shape and params"<<std::endl;
    x_q.print(2);
    x_k.print(2);
    x_v.print(2);
#endif

    M_Assert(embd_dim_head % 2 == 0);
    int embd_dim_head_complex = embd_dim_head / 2;

    std::vector<float> freqs_cis(embd_dim_head_complex); // [embd_vec_len]

    // implementation Q K RoPe
    for (int i = 0; i < embd_dim_head_complex; i++)
    {
        freqs_cis[i] = 1.0f / powf(10000.0f, i*2 / (float)(embd_dim_head));
    }

    std::vector<float > freqs_sin_cos(seq_len * embd_dim_head_complex * 2);

    for (int i = 0; i < seq_len; i++)
    {
        float* p_data = freqs_sin_cos.data() + i * embd_dim_head_complex * 2;

        int cur_seq = i + start_pos;
        for (int j = 0; j < embd_dim_head_complex; j++)
        {
            p_data[j*2] = sinf(cur_seq * freqs_cis[j]);
            p_data[j*2+1] = cosf(cur_seq * freqs_cis[j]);
        }
    }

    // freqs_cis = embd_vec_len * 2
    // kv shape is x_k = [bsz, seq, embd_dim_kv]
    for (int i = 0; i < seq_len; i++)
    {
        float* p_data = freqs_sin_cos.data() + i * embd_dim_head_complex * 2;
        float* p_x_q = (float *)x_q.data + i * embd_dim_head_complex * 2;
        float* p_x_k = (float *)x_k.data + i * embd_dim_head_complex * 2;

        for (int j = 0; j < embd_dim_head_complex; j++)
        {
            float freqs_sin = p_data[j*2];
            float freqs_cos = p_data[j*2 + 1];

            float q_r = p_x_q[j*2];
            float q_i = p_x_q[j*2+1];
            p_x_q[j*2]     = q_r * freqs_cos - q_i * freqs_sin;
            p_x_q[j*2 + 1] = q_r * freqs_sin + q_i * freqs_cos;

            float k_r = p_x_k[j*2];
            float k_i = p_x_k[j*2+1];
            p_x_k[j*2]     = k_r * freqs_cos - k_i * freqs_sin;
            p_x_k[j*2 + 1] = k_r * freqs_sin + k_i * freqs_cos;
        }
    }

    // TODO repeat kv based on repeat

    // implementation Q K matmul and mask
    Mat qk = gemm(x_q, x_k, false, true); // qk shape is [bsz, seq_len, seq_len + cache_len]

    int v = 4;
    uint32_t uy = 0;
    uint8_t data[4];
    float d = 0;
    uint32_t da = 0;
    char* dd = (char*)&d;
    memcpy(&uy, &v, sizeof(float ));
    memset(&da, uy, sizeof(float ));

    std::cout<<"data = "<<da<<std::endl;
    da = uy;
    std::cout<<"data = "<<da<<std::endl;

    Mat b = Mat({2}, qk.type());
    b.setTo(1);
    b.print();

    // implementation softmax
    Mat qk_sqrt = qk / sqrtf(embd_dim_head);

#if ATTEN_DEBUG
    std::cout<<"print after rope xq, xk, qk, qk_sqrt shape and params"<<std::endl;
    x_q.print(1);
    x_k.print(1);
    qk.print(2);
    qk_sqrt.print(2);
    std::cout<<"sqrtf(embd_dim_head) = "<<sqrtf(embd_dim_head)<<std::endl;
#endif

    // construct Mask Mat
    int dim_qk = qk_sqrt.size.dims();
    M_Assert(dim_qk >= 2);
    std::vector<int> mask_shape(dim_qk, 1);
    size_t m = qk_sqrt.size.p[dim_qk - 2];
    size_t n = qk_sqrt.size.p[dim_qk - 1];

    mask_shape[dim_qk - 1] = qk_sqrt.size.p[dim_qk - 1];
    mask_shape[dim_qk - 2] = qk_sqrt.size.p[dim_qk - 2];

    Mat mask = Mat(mask_shape, DT_32F);
    float* p_mask = (float*)mask.data;
    for (int i = 0; i < m; i++)
    {
        float* p_mask_i = p_mask + i*n;
        for (int j = 0; j < n; j++)
        {
            p_mask_i[j] = i >= j ? 1 : 0;
        }
    }

    Mat mask_1e20 = (mask - 1) * 1e20f;

    qk_sqrt = qk_sqrt * mask - mask_1e20;

    // apply the softmax to score
    Mat score = softmax(qk_sqrt);

#if ATTEN_DEBUG
        std::cout<<"print after mask, print qk_sqrt, score and x_v shape and params"<<std::endl;
        qk_sqrt.print(2);
        score.print(2);
        x_v.print(2);
#endif
    // implementation matmul V
    Mat qkv = gemm(score, x_v); // qk shape is [bsz, seq_len, seq_len + cache_len]

    // implementation out linear.
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims() - 1, out.size.p+1, out.type(), out.data);
    gemm(qkv, wout).copyTo(x_out);

    Mat m2 = out + *input[0];
    m2.copyTo(out);
}

void precompute_freq_cis(int dim, int end, int rms_eps)
{

}

void AttentionLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{

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

