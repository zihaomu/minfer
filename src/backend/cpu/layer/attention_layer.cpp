//
// Created by moo on 2024/8/4.
//

#include "attention_layer.h"
#include "autobuffer.h"
#include "backend/cpu/kernel/normalization_kernel_xsimd.h"
#include <cstring>  // for memcpy

#define ATTEN_DEBUG 0
namespace minfer {

namespace {

Mat project_with_runtime_weight(const Mat& input, const RuntimeWeight& weight)
{
    return weight.gemmNT(input);
}

Mat rmsnorm_with_runtime_weight(const Mat& input, const RuntimeWeight& weight, float eps)
{
    if (weight.usesInt8())
    {
        return rmsnorm(input, weight.active(), weight.int8Scales(), eps);
    }
    return rmsnorm(input, weight.active(), eps);
}

}  // namespace

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
    getBasicInfo(param);
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

    norm.init(param->norm, Int8QuantScheme::PerTensor);
    wq.init(canonicalize_linear_weight(param->wq, embd_dim, embd_dim), Int8QuantScheme::PerRow, true);
    wk.init(canonicalize_linear_weight(param->wk, embd_dim_kv, embd_dim), Int8QuantScheme::PerRow, true);
    wv.init(canonicalize_linear_weight(param->wv, embd_dim_kv, embd_dim), Int8QuantScheme::PerRow, true);
    wout.init(canonicalize_linear_weight(param->wout, embd_dim, embd_dim), Int8QuantScheme::PerRow, true);

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

struct QKVHeads {
    Mat q; // [seq_len, head_count, embd_dim_head]
    Mat k; // [seq_len, head_count_kv, embd_dim_head]
    Mat v; // [seq_len, head_count_kv, embd_dim_head]
};

static QKVHeads compute_qkv_heads(const Mat& x,
                                  const RuntimeWeight& norm,
                                  const RuntimeWeight& wq,
                                  const RuntimeWeight& wk,
                                  const RuntimeWeight& wv,
                                  int seq_len,
                                  int start_pos,
                                  int embd_dim,
                                  float rms_eps,
                                  int head_count,
                                  int head_count_kv,
                                  int embd_dim_head)
{
    Mat x_rows = Mat(x.dims - 1, x.size.p + 1, DT_32F, x.data);
    Mat x_norm = rmsnorm_with_runtime_weight(x_rows, norm, rms_eps);

    Mat x_q = project_with_runtime_weight(x_norm, wq);
    Mat x_k = project_with_runtime_weight(x_norm, wk);
    Mat x_v = project_with_runtime_weight(x_norm, wv);

    M_Assert(embd_dim_head % 2 == 0);
    x_q = x_q.reshape({seq_len, head_count, embd_dim_head});
    x_k = x_k.reshape({seq_len, head_count_kv, embd_dim_head});
    x_v = x_v.reshape({seq_len, head_count_kv, embd_dim_head});
    rope(x_q, x_k, start_pos);

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
                                            const RuntimeWeight& wout,
                                            int seq_len,
                                            int head_count,
                                            int embd_dim_head,
                                            const Mat& residual,
                                            Mat& out)
{
    Mat x_out = Mat(out.size.dims() - 1, out.size.p + 1, out.type(), out.data);
    Mat qkv_t = transposeND(qkv, {1, 0, 2});
    qkv_t = qkv_t.reshape({seq_len, head_count * embd_dim_head});
    project_with_runtime_weight(qkv_t, wout).copyTo(x_out);
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
    const size_t softmax_outer = qk.total() / (static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len));
    cpu::causal_masked_softmax_square_xsimd(reinterpret_cast<const float*>(qk.data),
                                            reinterpret_cast<float*>(qk.data),
                                            softmax_outer,
                                            seq_len,
                                            1.0f / sqrtf(embd_dim_head));

    // Step 8: score * V
    Mat attn_out = gemm(qk, x_v);

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

void AttentionLayer::setRuntimePrecision(RuntimePrecision precision)
{
    Layer::setRuntimePrecision(precision);
    norm.setPrecision(precision);
    wq.setPrecision(precision);
    wk.setPrecision(precision);
    wv.setPrecision(precision);
    wout.setPrecision(precision);
}

}
