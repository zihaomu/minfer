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
    M_Assert(input.size() == 1 && input[0]);
    MatShape in_shape = input[0]->shape();
    M_Assert(in_shape.size() == 3);

    InferenceContext ctx;
    ctx.phase = InferPhase::Prefill;
    ctx.start_pos = start_pos;
    ctx.seq_len = in_shape[1];

    forwardPrefill(input, output, ctx);
    start_pos += ctx.seq_len;
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

struct QKVHeads {
    Mat q; // [seq_len, head_count, embd_dim_head]
    Mat k; // [seq_len, head_count_kv, embd_dim_head]
    Mat v; // [seq_len, head_count_kv, embd_dim_head]
};

static QKVHeads compute_qkv_heads(const Mat& x,
                                  const Mat& norm,
                                  const Mat& wq,
                                  const Mat& wk,
                                  const Mat& wv,
                                  int seq_len,
                                  int start_pos,
                                  int embd_dim,
                                  float rms_eps,
                                  int head_count,
                                  int head_count_kv,
                                  int embd_dim_head)
{
    Mat x_norm = Mat(x.dims - 1, x.size.p + 1, DT_32F);
    rms_norm_slice((float*)x.data, (float*)x_norm.data, (float*)norm.data, seq_len, embd_dim, rms_eps);

    Mat x_q = gemm(x_norm, wq, false, true);
    Mat x_k = gemm(x_norm, wk, false, true);
    Mat x_v = gemm(x_norm, wv, false, true);

    M_Assert(embd_dim_head % 2 == 0);
    apply_rope((float*)x_q.data, (float*)x_k.data, seq_len, start_pos, head_count, head_count_kv, embd_dim_head);

    x_q = x_q.reshape({seq_len, head_count, embd_dim_head});
    x_k = x_k.reshape({seq_len, head_count_kv, embd_dim_head});
    x_v = x_v.reshape({seq_len, head_count_kv, embd_dim_head});

    return {x_q, x_k, x_v};
}

static void repeat_kv_if_needed(Mat& x_k,
                                Mat& x_v,
                                int seq_len,
                                int head_count,
                                int head_count_kv,
                                int embd_dim_head,
                                int repeat_kv)
{
    if (repeat_kv <= 1) return;

    Mat x_k_repeated({seq_len, head_count, embd_dim_head}, x_k.type());
    Mat x_v_repeated({seq_len, head_count, embd_dim_head}, x_v.type());

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

static void project_output_and_add_residual(const Mat& qkv,
                                            const Mat& wout,
                                            int seq_len,
                                            int head_count,
                                            int embd_dim_head,
                                            const Mat& residual,
                                            Mat& out)
{
    Mat x_out = Mat(out.size.dims() - 1, out.size.p + 1, out.type(), out.data);
    Mat qkv_t = transposeND(qkv, {1, 0, 2});
    qkv_t = qkv_t.reshape({seq_len, head_count * embd_dim_head});
    gemm(qkv_t, wout, false, true).copyTo(x_out);
    out += residual;
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

    Mat x = *input[0];
    QKVHeads qkv_heads = compute_qkv_heads(
        x, norm, wq, wk, wv, seq_len, ctx.start_pos,
        embd_dim, rms_eps, head_count, head_count_kv, embd_dim_head);
    Mat x_q = qkv_heads.q;
    Mat x_k = qkv_heads.k;
    Mat x_v = qkv_heads.v;

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
    repeat_kv_if_needed(x_k, x_v, seq_len, head_count, head_count_kv, embd_dim_head, repeat_kv);

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
    Mat attn_out = gemm(score, x_v);

    // Step 9: Transpose back and output linear
    Mat out = *output[0];
    project_output_and_add_residual(attn_out, wout, seq_len, head_count, embd_dim_head, x, out);
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

    Mat x = *input[0];
    QKVHeads qkv_heads = compute_qkv_heads(
        x, norm, wq, wk, wv, 1, cur_pos,
        embd_dim, rms_eps, head_count, head_count_kv, embd_dim_head);
    Mat x_q = qkv_heads.q;
    Mat x_k = qkv_heads.k;
    Mat x_v = qkv_heads.v;

    // Step 3: Write new K, V to cache at position cur_pos
    // x_k/x_v shape: [1, head_count_kv, embd_dim_head]
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
    Mat attn_out = gemm(score, v_attn);

    Mat out = *output[0];
    project_output_and_add_residual(attn_out, wout, 1, head_count, embd_dim_head, x, out);
}

void AttentionLayer::resetKVCache()
{
    k_cache.setTo(0.0f);
    v_cache.setTo(0.0f);
    cached_len = 0;
    start_pos = 0;
}

}
