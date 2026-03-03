//
// Created by moo on 2024/8/4.
//

#include "attention_layer.h"
#include "autobuffer.h"
#include <cstring>  // for memcpy

#define ATTEN_DEBUG 0
namespace minfer {

#if ATTEN_DEBUG
void print_mat(const Mat& m, int start, int num)
{
    const float* p = (const float*)m.data;
    for (int i = start; i < start + num; i++)
    {
        std::cout<<p[i]<<" ";
    }
    std::cout<<std::endl;
}
#endif

AttentionLayer::AttentionLayer(const std::shared_ptr<AttentionLayerParams> param)
{
    layerNamePrefix = "AttentionLayer_";
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

    std::cout << "DEBUG: Attn Layer Init Q: [" << wq.size[0] << ", " << wq.size[1] << "]" << std::endl;
    std::cout << "DEBUG: Attn Layer Init K: [" << wk.size[0] << ", " << wk.size[1] << "]" << std::endl;
    std::cout << "DEBUG: Attn Layer Init V: [" << wv.size[0] << ", " << wv.size[1] << "]" << std::endl;
    std::cout << "DEBUG: Attn Layer Init Out: [" << wout.size[0] << ", " << wout.size[1] << "]" << std::endl;

    if (layerNamePrefix == "AttentionLayer_") {
        float* wq_ptr = (float*)wq.data;
        std::cout << "DEBUG: Attn WQ first 5: ";
        for(int k=0; k<5; k++) std::cout << wq_ptr[k] << " ";
        std::cout << "\nDEBUG: Attn WQ last 5: ";
        for(int k=wq.total()-5; k<wq.total(); k++) std::cout << wq_ptr[k] << " ";
        std::cout << std::endl;
    }

#if ATTEN_DEBUG
    std::cout<<"print in init q k v out shape and params"<<std::endl;
    wq.print(2);
    wk.print(2);
    wv.print(2);
    wout.print(2);
#endif

    // minfer::gemm does not currently support `transposeB=true`.
    // GGUF K and V matrices are stored as `[embd_dim_kv, embd_dim]`, so we
    // transpose them here during initialization so that `gemm` with `false, false` works.

    // 预分配 KV Cache
    std::vector<int> cache_shape = {head_count_kv, max_seq_len, embd_dim_head};
    k_cache = Mat(cache_shape, DT_32F, 0);
    v_cache = Mat(cache_shape, DT_32F, 0);
    cached_len = 0;
}

void AttentionLayer::finalize(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{

}

Mat softmax(Mat inp)
{
    Mat out = inp.clone();
    M_Assert(inp.type() == DT_32F);

    int inp_dim = inp.dims;

    M_Assert(inp_dim > 1);

    size_t last_len = inp.size[inp_dim - 1];
    size_t out_loop = inp.total(0, inp_dim - 1);

    for (int l = 0; l < out_loop; l++)
    {
        const float* p_i = (const float*)inp.data + last_len * l;
        float* p_o = (float*)out.data + last_len * l;

        float max_val = *std::max_element(p_i, p_i + last_len);

        float sum = 0.0f;
        for (int i = 0; i < last_len; i++)
        {
            p_o[i] = expf(p_i[i] - max_val);
            sum += p_o[i];
        }

        // normalize
        float sum_div = 1.f/sum;
        for (int i = 0; i < last_len; i++)
        {
            p_o[i] *= sum_div;
        }
    }

    return out;
}

/* forward function contains two operator, RMSnorm and attention.
 * forward contain start_pos and sequence len, how to set the sequence len to the forward?
 * */
// TODO take into account the kv_head is different with head_count.
// TODO try to use bias params
void AttentionLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output)
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

    Mat x_q = gemm(x_norm, wq, false, true); 
    Mat x_k = gemm(x_norm, wk, false, true); 
    Mat x_v = gemm(x_norm, wv, false, true);

#if ATTEN_DEBUG
    std::cout<<"print x_norm"<<std::endl;
    x_norm.print(10);
    std::cout<<"print in init xq, xk xv shape and params"<<std::endl;
    x_q.print(20);
    x_k.print(20);
    x_v.print(20);

    std::cout<<"print_mat "<<std::endl;
    print_mat(x_q, 128, 20);
    print_mat(x_k, 128, 20);
    print_mat(x_v, 128, 20);

#endif

    M_Assert(embd_dim_head % 2 == 0);
    int embd_dim_head_complex = embd_dim_head / 2;

    std::vector<float> freqs_cis(embd_dim_head_complex); // [embd_vec_len]

    // implementation Q K RoPe
    for (int i = 0; i < embd_dim_head_complex; i++)
    {
        freqs_cis[i] = 1.0f / powf(10000.0f, i*2 / (float)(embd_dim_head));
    }

    std::vector<float > freqs_sin_cos(seq_len * embd_dim_head_complex * 2); // [seq_len, embd_dim_head_complex, 2]

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

#if ATTEN_DEBUG
    // debug freqs_sin_cos
    if (seq_len > 1) {
        float* p_data_t1 = freqs_sin_cos.data() + embd_dim_head_complex * 2;
        std::cout<<"DEBUG: Token 1 freqs sin = ";
        for (int i = 0; i < 10; i++)
        {
            std::cout<<p_data_t1[i*2]<<",";
        }
        std::cout<<std::endl;

        std::cout<<"DEBUG: Token 1 freqs cos = ";
        for (int i = 0; i < 10; i++)
        {
            std::cout<<p_data_t1[i*2 + 1]<<",";
        }
        std::cout<<std::endl;
    }
#endif

    // freqs_cis = embd_vec_len * 2
    // kv shape is x_k = [bsz, seq, embd_dim_kv]
    // Debug this part code.
    for (int i = 0; i < seq_len; i++) // seq
    {
        float* p_data = freqs_sin_cos.data() + i * embd_dim_head_complex * 2;
        float* p_x_q = (float *)x_q.data + i * embd_dim_head_complex * 2 * head_count;
        float* p_x_k = (float *)x_k.data + i * embd_dim_head_complex * 2 * head_count_kv;

        // Apply RoPE to Q
        for (int h = 0; h < head_count; h++) // head
        {
            for (int j = 0; j < embd_dim_head_complex; j++) // internal
            {
                float freqs_sin = p_data[j*2];
                float freqs_cos = p_data[j*2 + 1];

                float q_r = p_x_q[j*2];
                float q_i = p_x_q[j*2+1];
                p_x_q[j*2]     = q_r * freqs_cos - q_i * freqs_sin;
                p_x_q[j*2 + 1] = q_r * freqs_sin + q_i * freqs_cos;
            }
            p_x_q += embd_dim_head_complex * 2;
        }

        // Apply RoPE to K (using head_count_kv)
        for (int h = 0; h < head_count_kv; h++) // head_kv
        {
            for (int j = 0; j < embd_dim_head_complex; j++) // internal
            {
                float freqs_sin = p_data[j*2];
                float freqs_cos = p_data[j*2 + 1];

                float k_r = p_x_k[j*2];
                float k_i = p_x_k[j*2+1];
                p_x_k[j*2]     = k_r * freqs_cos - k_i * freqs_sin;
                p_x_k[j*2 + 1] = k_r * freqs_sin + k_i * freqs_cos;
            }
            p_x_k += embd_dim_head_complex * 2;
        }
    }

    // TODO repeat kv based on repeat

#if ATTEN_DEBUG
    std::cout<<"print after RoPE xq, xk, xv shape and params"<<std::endl;
    x_q.print(10);
    x_k.print(10);

    std::cout<<"print_mat "<<std::endl;
    print_mat(x_q, 128, 20);
    print_mat(x_k, 128, 20);
#endif

    // because x_q, x_k, x_v are two dim, we need to reshape it to [seq, bsz, embd_dim_head]
    // TODO check if we should use head_count_kv?
    std::vector<int> new_shape_q = {seq_len, head_count, embd_dim_head};
    std::vector<int> new_shape_kv = {seq_len, head_count_kv, embd_dim_head};

    x_q = x_q.reshape(new_shape_q);
    x_k = x_k.reshape(new_shape_kv);
    x_v = x_v.reshape(new_shape_kv);

#if ATTEN_DEBUG
    if (layerNamePrefix == "AttentionLayer_") {
        float* xk_ptr = (float*)x_k.data;
        std::cout << "DEBUG: Attn Layer 0 Token 0 computed x_k First 10 (reshaped): ";
        for(int k=0; k<10; k++) std::cout << xk_ptr[k] << " ";
        std::cout << std::endl;
    }
#endif

    // Repeat K and V to match Q's head count for Grouped Query Attention
    if (repeat_kv > 1) {
        // Need to repeat K and V along the head dimension
        // x_k shape: [seq_len, head_count_kv, embd_dim_head] -> [seq_len, head_count, embd_dim_head]
        // x_v shape: [seq_len, head_count_kv, embd_dim_head] -> [seq_len, head_count, embd_dim_head]
        
        Mat x_k_repeated = Mat({seq_len, head_count, embd_dim_head}, x_k.type());
        Mat x_v_repeated = Mat({seq_len, head_count, embd_dim_head}, x_v.type());
        
        float* k_src = (float*)x_k.data;
        float* v_src = (float*)x_v.data;
        float* k_dst = (float*)x_k_repeated.data;
        float* v_dst = (float*)x_v_repeated.data;
        
        for (int s = 0; s < seq_len; s++) {
            for (int h = 0; h < head_count; h++) {
                int kv_head = h / repeat_kv; // Which kv head to copy from
                
                // Copy K
                memcpy(k_dst + (s * head_count + h) * embd_dim_head,
                       k_src + (s * head_count_kv + kv_head) * embd_dim_head,
                       embd_dim_head * sizeof(float));
                
                // Copy V
                memcpy(v_dst + (s * head_count + h) * embd_dim_head,
                       v_src + (s * head_count_kv + kv_head) * embd_dim_head,
                       embd_dim_head * sizeof(float));
            }
        }
        
        x_k = x_k_repeated;
        x_v = x_v_repeated;
    }

    // py code: scores = self.attn.forward(q, k, v)
    x_q = transposeND(x_q, {1, 0, 2});
    x_k = transposeND(x_k, {1, 0, 2});
    x_v = transposeND(x_v, {1, 0, 2});

    // implementation Q K matmul and mask
    Mat qk = gemm(x_q, x_k, false, true); // qk shape is [bsz, seq_len, seq_len + cache_len]

    // implementation softmax
    Mat qk_sqrt = qk / sqrtf(embd_dim_head); // score

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

    // std::cout<<"before print_mat mask  mask_1e20"<<std::endl;
    // mask.print(10);
    // print_mat(mask, 0, 20);
    // print_mat(mask, 256, 20);
    // print_mat(mask, 256 * 2, 20);

    Mat mask_1e20 = (1.f - mask) * 1e20f;

    // std::cout<<"print_mat mask  mask_1e20"<<std::endl;
    // mask_1e20.print(10);
    // print_mat(mask_1e20, 0, 20);
    // print_mat(mask_1e20, 256, 20);
    // print_mat(mask_1e20, 256 * 2, 20);
    // qk_sqrt.print(10);
    // 8 x 256 x 256
    qk_sqrt = qk_sqrt * mask - mask_1e20;

    // std::cout<<"print_mat mask "<<std::endl;
    // print_mat(qk_sqrt, 0, 20);
    // print_mat(qk_sqrt, 256*256, 20);
    // print_mat(qk_sqrt,  256*256*2, 20);
    //
    // apply the softmax to score
    Mat score = softmax(qk_sqrt);
    qk_sqrt.print(2);
    score.print(2);
    x_v.print(2);
    // implementation matmul V
    Mat qkv = gemm(score, x_v); // qk shape is [bsz, seq_len, seq_len + cache_len]
    // implementation out linear.
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims() - 1, out.size.p+1, out.type(), out.data);

    // Transpose qkv from [head_count, seq_len, embd_dim_head] to [seq_len, head_count, embd_dim_head]
    // and then reshape it to [seq_len, head_count * embd_dim_head]
    Mat qkvT= transposeND(qkv, {1, 0, 2});
    qkvT = qkvT.reshape({seq_len, head_count * embd_dim_head});
    // std::cout<<"qkvT.print("<<std::endl;
    // qkvT.print(20);
    // print_mat(qkvT, 128, 20);
    // print_mat(qkvT, 128*2, 20);
    // print_mat(qkvT, 128*3, 20);
    // print_mat(qkvT, 128*4, 20);
    // wout.print(20);
    // print_mat(wout, 128, 20);
    // print_mat(wout, 128*2, 20);
    gemm(qkvT, wout, false, true).copyTo(x_out);
    //
    // std::cout<<"out, 0"<<std::endl;
    // print_mat(out, 0, 20);
    // print_mat(out, 128, 20);
    // print_mat(out, 128 * 2, 20);
    // print_mat(out, 128 * 3, 20);
    // Residual connection (就地加法，确保结果写回 *output[0])
    out += *input[0];

    // 最后加上这次的seq len
    start_pos += seq_len;
}

void precompute_freq_cis(int dim, int end, int rms_eps)
{

}

void AttentionLayer::init(const std::vector<Mat *> &input, std::vector<Mat *> &output)
{
    // pre check
    int input_num = input.size();

    M_Assert(input_num == 1);
    M_Assert(output.size() == 1);

    // 设置同样的shape
    output[0]->setSize(*input[0]);
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

// ====== Chat Forward with InferenceContext ======

void AttentionLayer::forward(const std::vector<Mat *> &input, std::vector<Mat *> &output, const InferenceContext& ctx)
{
    if (ctx.phase == InferPhase::Prefill)
    {
        forwardPrefill(input, output, ctx);
    }
    else
    {
        forwardDecode(input, output, ctx);
    }
}

// Helper: compute RMS norm for a slice of tokens
static void rms_norm_slice(const float* input, float* out, const float* norm_w, int seq_len, int embd_dim, float eps)
{
    for (int i = 0; i < seq_len; i++)
    {
        const float* pi_s = input + i * embd_dim;
        float* po = out + i * embd_dim;

        float sum_f2 = 0;
        for (int j = 0; j < embd_dim; j++)
            sum_f2 += pi_s[j] * pi_s[j];

        float x1 = 1.f / sqrtf(sum_f2 / embd_dim + eps);
        for (int j = 0; j < embd_dim; j++)
            po[j] = pi_s[j] * x1 * norm_w[j];
    }
}

// Helper: apply RoPE to Q and K
static void apply_rope(float* x_q_data, float* x_k_data, int seq_len, int start_pos,
                        int head_count, int head_count_kv, int embd_dim_head)
{
    int embd_dim_head_complex = embd_dim_head / 2;
    std::vector<float> freqs_cis(embd_dim_head_complex);
    for (int i = 0; i < embd_dim_head_complex; i++)
        freqs_cis[i] = 1.0f / powf(10000.0f, i * 2 / (float)(embd_dim_head));

    for (int i = 0; i < seq_len; i++)
    {
        int cur_seq = i + start_pos;
        float* p_x_q = x_q_data + i * embd_dim_head * head_count;
        float* p_x_k = x_k_data + i * embd_dim_head * head_count_kv;

        for (int h = 0; h < head_count; h++)
        {
            for (int j = 0; j < embd_dim_head_complex; j++)
            {
                float freqs_sin = sinf(cur_seq * freqs_cis[j]);
                float freqs_cos = cosf(cur_seq * freqs_cis[j]);
                float q_r = p_x_q[j * 2];
                float q_i = p_x_q[j * 2 + 1];
                p_x_q[j * 2]     = q_r * freqs_cos - q_i * freqs_sin;
                p_x_q[j * 2 + 1] = q_r * freqs_sin + q_i * freqs_cos;
            }
            p_x_q += embd_dim_head;
        }

        for (int h = 0; h < head_count_kv; h++)
        {
            for (int j = 0; j < embd_dim_head_complex; j++)
            {
                float freqs_sin = sinf(cur_seq * freqs_cis[j]);
                float freqs_cos = cosf(cur_seq * freqs_cis[j]);
                float k_r = p_x_k[j * 2];
                float k_i = p_x_k[j * 2 + 1];
                p_x_k[j * 2]     = k_r * freqs_cos - k_i * freqs_sin;
                p_x_k[j * 2 + 1] = k_r * freqs_sin + k_i * freqs_cos;
            }
            p_x_k += embd_dim_head;
        }
    }
}

void AttentionLayer::forwardPrefill(const std::vector<Mat *> &input, std::vector<Mat *> &output, const InferenceContext& ctx)
{
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[2] == embd_dim);
    M_Assert(in_shape[0] == 1 && "Currently, only support single batch!");
    M_Assert(input[0]->type() == DT_32F);

    int seq_len = in_shape[1];

    // Step 0: RMS Norm
    Mat x = *input[0];
    Mat x_norm = Mat(x.dims - 1, x.size.p + 1, DT_32F);
    rms_norm_slice((float*)x.data, (float*)x_norm.data, (float*)norm.data, seq_len, embd_dim, rms_eps);

    // Step 1: Q K V linear
    Mat x_q = gemm(x_norm, wq, false, true);
    Mat x_k = gemm(x_norm, wk, false, true);
    Mat x_v = gemm(x_norm, wv, false, true);

    // Step 2: RoPE
    M_Assert(embd_dim_head % 2 == 0);
    apply_rope((float*)x_q.data, (float*)x_k.data, seq_len, ctx.start_pos,
               head_count, head_count_kv, embd_dim_head);

    // Step 3: Reshape to multi-head format
    std::vector<int> new_shape_q = {seq_len, head_count, embd_dim_head};
    std::vector<int> new_shape_kv = {seq_len, head_count_kv, embd_dim_head};
    x_q = x_q.reshape(new_shape_q);
    x_k = x_k.reshape(new_shape_kv);
    x_v = x_v.reshape(new_shape_kv);

    // Step 4: Write K, V to cache
    // k_cache/v_cache shape: [head_count_kv, max_seq_len, embd_dim_head]
    // x_k shape: [seq_len, head_count_kv, embd_dim_head] -> need to transpose to [head_count_kv, seq_len, embd_dim_head]
    Mat x_k_t = transposeND(x_k, {1, 0, 2}); // [head_count_kv, seq_len, embd_dim_head]
    Mat x_v_t = transposeND(x_v, {1, 0, 2});

    for (int h = 0; h < head_count_kv; h++)
    {
        float* dst_k = (float*)k_cache.data + h * max_seq_len * embd_dim_head + ctx.start_pos * embd_dim_head;
        float* dst_v = (float*)v_cache.data + h * max_seq_len * embd_dim_head + ctx.start_pos * embd_dim_head;
        float* src_k = (float*)x_k_t.data + h * seq_len * embd_dim_head;
        float* src_v = (float*)x_v_t.data + h * seq_len * embd_dim_head;
        memcpy(dst_k, src_k, seq_len * embd_dim_head * sizeof(float));
        memcpy(dst_v, src_v, seq_len * embd_dim_head * sizeof(float));
    }
    cached_len = ctx.start_pos + seq_len;

    // Step 5: Repeat KV for GQA
    if (repeat_kv > 1)
    {
        Mat x_k_repeated = Mat({seq_len, head_count, embd_dim_head}, x_k.type());
        Mat x_v_repeated = Mat({seq_len, head_count, embd_dim_head}, x_v.type());
        float* k_src = (float*)x_k.data;
        float* v_src = (float*)x_v.data;
        float* k_dst = (float*)x_k_repeated.data;
        float* v_dst = (float*)x_v_repeated.data;
        for (int s = 0; s < seq_len; s++)
        {
            for (int h = 0; h < head_count; h++)
            {
                int kv_head = h / repeat_kv;
                memcpy(k_dst + (s * head_count + h) * embd_dim_head,
                       k_src + (s * head_count_kv + kv_head) * embd_dim_head,
                       embd_dim_head * sizeof(float));
                memcpy(v_dst + (s * head_count + h) * embd_dim_head,
                       v_src + (s * head_count_kv + kv_head) * embd_dim_head,
                       embd_dim_head * sizeof(float));
            }
        }
        x_k = x_k_repeated;
        x_v = x_v_repeated;
    }

    // Step 6: Transpose for attention
    x_q = transposeND(x_q, {1, 0, 2}); // [head_count, seq_len, embd_dim_head]
    x_k = transposeND(x_k, {1, 0, 2});
    x_v = transposeND(x_v, {1, 0, 2});

    // Step 7: Attention with causal mask
    Mat qk = gemm(x_q, x_k, false, true);
    Mat qk_sqrt = qk / sqrtf(embd_dim_head);

    // Build causal mask
    int dim_qk = qk_sqrt.size.dims();
    size_t m = qk_sqrt.size.p[dim_qk - 2];
    size_t n = qk_sqrt.size.p[dim_qk - 1];
    std::vector<int> mask_shape(dim_qk, 1);
    mask_shape[dim_qk - 1] = n;
    mask_shape[dim_qk - 2] = m;
    Mat mask = Mat(mask_shape, DT_32F);
    float* p_mask = (float*)mask.data;
    for (int i = 0; i < (int)m; i++)
        for (int j = 0; j < (int)n; j++)
            p_mask[i * n + j] = i >= j ? 1.0f : 0.0f;

    Mat mask_1e20 = (1.f - mask) * 1e20f;
    qk_sqrt = qk_sqrt * mask - mask_1e20;

    Mat score = softmax(qk_sqrt);

    // Step 8: score * V
    Mat qkv = gemm(score, x_v);

    // Step 9: Transpose back and output linear
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims() - 1, out.size.p + 1, out.type(), out.data);
    Mat qkvT = transposeND(qkv, {1, 0, 2});
    qkvT = qkvT.reshape({seq_len, head_count * embd_dim_head});
    gemm(qkvT, wout, false, true).copyTo(x_out);

    // Residual connection
    // Residual connection (就地加法，确保结果写回 *output[0])
    out += *input[0];
}

void AttentionLayer::forwardDecode(const std::vector<Mat *> &input, std::vector<Mat *> &output, const InferenceContext& ctx)
{
    M_Assert(input.size() == 1 && input[0]);
    M_Assert(output.size() == 1 && output[0]);

    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 3);
    M_Assert(in_shape[1] == 1 && "Decode phase should process 1 token at a time!");
    M_Assert(in_shape[2] == embd_dim);
    M_Assert(input[0]->type() == DT_32F);

    int cur_pos = ctx.start_pos; // position of this new token

    // Step 0: RMS Norm (1 token)
    Mat x = *input[0];
    Mat x_norm = Mat(x.dims - 1, x.size.p + 1, DT_32F);
    rms_norm_slice((float*)x.data, (float*)x_norm.data, (float*)norm.data, 1, embd_dim, rms_eps);

    // Step 1: Q K V linear (1 token)
    Mat x_q = gemm(x_norm, wq, false, true); // [1, embd_dim]
    Mat x_k = gemm(x_norm, wk, false, true); // [1, embd_dim_kv]
    Mat x_v = gemm(x_norm, wv, false, true);

    // Step 2: RoPE (1 token)
    M_Assert(embd_dim_head % 2 == 0);
    apply_rope((float*)x_q.data, (float*)x_k.data, 1, cur_pos,
               head_count, head_count_kv, embd_dim_head);

    // Step 3: Write new K, V to cache at position cur_pos
    // x_k shape: [1, embd_dim_kv] = [1, head_count_kv * embd_dim_head]
    for (int h = 0; h < head_count_kv; h++)
    {
        float* dst_k = (float*)k_cache.data + h * max_seq_len * embd_dim_head + cur_pos * embd_dim_head;
        float* dst_v = (float*)v_cache.data + h * max_seq_len * embd_dim_head + cur_pos * embd_dim_head;
        float* src_k = (float*)x_k.data + h * embd_dim_head;
        float* src_v = (float*)x_v.data + h * embd_dim_head;
        memcpy(dst_k, src_k, embd_dim_head * sizeof(float));
        memcpy(dst_v, src_v, embd_dim_head * sizeof(float));
    }
    cached_len = cur_pos + 1;

    int total_len = cached_len; // total KV sequence length including this token

    // Step 4: Reshape Q for multi-head: [1, head_count, embd_dim_head]
    x_q = x_q.reshape({1, head_count, embd_dim_head});

    // Step 5: Get full K, V from cache for attention: [head_count_kv, total_len, embd_dim_head]
    // Slice cache to [head_count_kv, total_len, embd_dim_head]
    Mat k_slice = Mat({head_count_kv, total_len, embd_dim_head}, DT_32F);
    Mat v_slice = Mat({head_count_kv, total_len, embd_dim_head}, DT_32F);
    for (int h = 0; h < head_count_kv; h++)
    {
        float* src_k = (float*)k_cache.data + h * max_seq_len * embd_dim_head;
        float* src_v = (float*)v_cache.data + h * max_seq_len * embd_dim_head;
        float* dst_k = (float*)k_slice.data + h * total_len * embd_dim_head;
        float* dst_v = (float*)v_slice.data + h * total_len * embd_dim_head;
        memcpy(dst_k, src_k, total_len * embd_dim_head * sizeof(float));
        memcpy(dst_v, src_v, total_len * embd_dim_head * sizeof(float));
    }

    // Step 6: Repeat KV for GQA, then transpose
    // k_slice: [head_count_kv, total_len, embd_dim_head]
    // Need to expand to [head_count, total_len, embd_dim_head] if GQA
    Mat k_attn, v_attn;
    if (repeat_kv > 1)
    {
        k_attn = Mat({head_count, total_len, embd_dim_head}, DT_32F);
        v_attn = Mat({head_count, total_len, embd_dim_head}, DT_32F);
        for (int h = 0; h < head_count; h++)
        {
            int kv_head = h / repeat_kv;
            memcpy((float*)k_attn.data + h * total_len * embd_dim_head,
                   (float*)k_slice.data + kv_head * total_len * embd_dim_head,
                   total_len * embd_dim_head * sizeof(float));
            memcpy((float*)v_attn.data + h * total_len * embd_dim_head,
                   (float*)v_slice.data + kv_head * total_len * embd_dim_head,
                   total_len * embd_dim_head * sizeof(float));
        }
    }
    else
    {
        k_attn = k_slice;
        v_attn = v_slice;
    }

    // Step 7: Compute attention
    // Q: [1, head_count, embd_dim_head] -> transpose to [head_count, 1, embd_dim_head]
    Mat q_t = transposeND(x_q, {1, 0, 2}); // [head_count, 1, embd_dim_head]
    // K: [head_count, total_len, embd_dim_head] (already in right format)

    // QK: [head_count, 1, total_len]
    Mat qk = gemm(q_t, k_attn, false, true);
    Mat qk_sqrt = qk / sqrtf(embd_dim_head);

    // Decode phase: single query token attends to all cached tokens, no mask needed
    Mat score = softmax(qk_sqrt);

    // Step 8: score * V: [head_count, 1, embd_dim_head]
    Mat qkv = gemm(score, v_attn);

    // Step 9: Transpose back and output linear
    Mat out = *output[0];
    Mat x_out = Mat(out.size.dims() - 1, out.size.p + 1, out.type(), out.data);
    Mat qkvT = transposeND(qkv, {1, 0, 2}); // [1, head_count, embd_dim_head]
    qkvT = qkvT.reshape({1, head_count * embd_dim_head});
    gemm(qkvT, wout, false, true).copyTo(x_out);

    // Residual connection
    // Residual connection (就地加法，确保结果写回 *output[0])
    out += *input[0];
}

void AttentionLayer::resetKVCache()
{
    k_cache.setTo(0.0f);
    v_cache.setTo(0.0f);
    cached_len = 0;
    start_pos = 0;
}

}

