//
// Created by moo on 2024/8/4.
//

#include "attention_layer.h"
#include "autobuffer.h"
#include "backend/cpu/kernel/normalization_kernel_xsimd.h"
#include "backend/cpu/kernel/openmp_utils.h"
#include "mobilekv/kv_cache.h"

#include <algorithm>
#include <cstring>  // for memcpy
#include <cmath>
#include <cfloat>
#include <cstdlib>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__) || defined(_M_ARM64)
#ifndef XSIMD_ENABLE_WASM
#define XSIMD_ENABLE_WASM 0
#endif
#endif
#include "xsimd/xsimd.hpp"

#define ATTEN_DEBUG 0
namespace minfer {

namespace {

inline float xsimd_dot_fp32(const float* a, const float* b, int N) {
    using batch_type = xsimd::batch<float>;
    int inc = batch_type::size;
    batch_type sum(0.0f);
    int i = 0;
    for (; i + inc <= N; i += inc) {
        sum = xsimd::fma(batch_type::load_unaligned(a + i), batch_type::load_unaligned(b + i), sum);
    }
    float res = xsimd::reduce_add(sum);
    for (; i < N; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

inline void xsimd_fmadd_inplace(float* out, float exp_diff, float exp_qk, const float* v, int N) {
    using batch_type = xsimd::batch<float>;
    int inc = batch_type::size;
    batch_type b_exp_diff(exp_diff);
    batch_type b_exp_qk(exp_qk);
    int i = 0;
    for (; i + inc <= N; i += inc) {
        batch_type b_out = batch_type::load_unaligned(out + i);
        batch_type b_v = batch_type::load_unaligned(v + i);
        batch_type res = xsimd::fma(b_v, b_exp_qk, b_out * b_exp_diff);
        res.store_unaligned(out + i);
    }
    for (; i < N; ++i) {
        out[i] = out[i] * exp_diff + exp_qk * v[i];
    }
}

Mat project_with_runtime_weight(const Mat& input, const RuntimeWeight& weight)
{
    return weight.gemmNT(input);
}

Mat rmsnorm_with_runtime_weight(const Mat& input, const RuntimeWeight& weight, float eps)
{
    Mat aligned_input = input;
    if (weight.precision() != RuntimePrecision::FP32)
    {
        aligned_input = align_precision_sensitive_input(input, weight.precision());
    }

    if (weight.usesInt8())
    {
        return rmsnorm(aligned_input, weight.active(), weight.int8Scales(), eps);
    }
    return rmsnorm(aligned_input, weight.active(), eps);
}

bool is_mobilekv_plain_fp32_compatible(const mobilekv::KVPlane& plane,
                                       int expected_heads,
                                       int expected_head_dim)
{
    const auto& cfg = plane.templ().config();
    const auto& shape = plane.templ().shape();
    return cfg.scalar_type == mobilekv::ScalarType::FP32 &&
           shape.num_heads == static_cast<uint32_t>(expected_heads) &&
           shape.head_dim == static_cast<uint32_t>(expected_head_dim);
}

void write_plain_fp32_tokens_to_mobilekv(mobilekv::KVPlane& plane,
                                         uint32_t layer_id,
                                         const float* src,
                                         int src_seq_len,
                                         int src_begin,
                                         uint32_t dst_begin,
                                         uint32_t token_count,
                                         int num_heads,
                                         int head_dim)
{
    M_Assert(src);
    uint8_t* base = static_cast<uint8_t*>(plane.data());
    M_Assert(base);

    const size_t token_stride = static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim);
    const size_t token_count_sz = static_cast<size_t>(token_count);
    const bool parallel_tokens = cpu::should_parallelize_1d_loop(
        token_count_sz,
        token_stride,
        1LL << 14,
        1);
    const long long token_count_ll = static_cast<long long>(token_count);
#ifdef _OPENMP
#pragma omp parallel for if(parallel_tokens)
#endif
    for (long long t_ll = 0; t_ll < token_count_ll; ++t_ll)
    {
        const uint32_t t = static_cast<uint32_t>(t_ll);
        const int src_t = src_begin + static_cast<int>(t);
        M_Assert(src_t >= 0 && src_t < src_seq_len);
        const uint32_t dst_t = dst_begin + t;
        for (int h = 0; h < num_heads; ++h)
        {
            const auto addr = plane.locate(mobilekv::LogicalCoord(layer_id, dst_t, static_cast<uint32_t>(h), 0));
            M_Assert(addr.valid);

            const float* src_ptr = src + (static_cast<size_t>(src_t) * token_stride +
                                          static_cast<size_t>(h) * static_cast<size_t>(head_dim));
            std::memcpy(base + addr.byte_offset, src_ptr, static_cast<size_t>(head_dim) * sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// Ring-buffer segment descriptor
// ---------------------------------------------------------------------------
// The PlainKVTemplate stores data in [S, H, D] (seq-major) order.
// When the ring-buffer has wrapped (write_head > 0) the valid tokens are
// split into two physically-contiguous slabs:
//   Segment A (older tokens): base[ write_head .. max_cap )  length = sa
//   Segment B (newer tokens): base[ 0          .. write_head) length = sb
//
// Logically the sequence is A followed by B.
// When not wrapped (write_head == 0) only segment B is non-empty.
// ---------------------------------------------------------------------------

struct RingSegments {
    const float* base;   // raw pointer to plane data()
    int sa;              // segment-A length  (older, high physical address)
    int sb;              // segment-B length  (newer, low  physical address)
    int write_head;      // physical start of segment B
    int num_heads;       // Hkv
    int head_dim;        // D
    int total;           // sa + sb
};

static RingSegments get_ring_segments(const mobilekv::KVPlane& plane)
{
    const auto& stats = plane.stats();
    const float* base = static_cast<const float*>(plane.data());
    M_Assert(base);

    RingSegments seg;
    seg.base      = base;
    seg.num_heads = static_cast<int>(plane.templ().shape().num_heads);
    seg.head_dim  = static_cast<int>(plane.templ().shape().head_dim);
    seg.total     = static_cast<int>(stats.seq_length);

    if (stats.is_ring_buffer && stats.max_seq_capacity > 0 &&
        stats.seq_length == stats.max_seq_capacity && stats.write_head > 0)
    {
        // Ring is full and has wrapped: valid data spans two physical slabs.
        //   Segment A (older): rows [write_head, max_cap)  length = sa
        //   Segment B (newer): rows [0,          write_head) length = sb
        seg.write_head = static_cast<int>(stats.write_head);
        seg.sa = static_cast<int>(stats.max_seq_capacity) - seg.write_head;
        seg.sb = seg.write_head;
    }
    else
    {
        // Ring is not full, or write_head has not yet wrapped.
        // All valid data is in one contiguous physical region [0, seq_length).
        seg.write_head = 0;
        seg.sa         = 0;
        seg.sb         = seg.total;
    }
    return seg;
}

// Gather one KV-head's data from a [S, Hkv, D] slab for a given segment.
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

    norm.init(param->norm, Int8QuantScheme::PerTensor, false, param->precision);
    wq.init(canonicalize_linear_weight(param->wq, embd_dim, embd_dim), Int8QuantScheme::PerRow, true, param->precision);
    wk.init(canonicalize_linear_weight(param->wk, embd_dim_kv, embd_dim), Int8QuantScheme::PerRow, true, param->precision);
    wv.init(canonicalize_linear_weight(param->wv, embd_dim_kv, embd_dim), Int8QuantScheme::PerRow, true, param->precision);
    wout.init(canonicalize_linear_weight(param->wout, embd_dim, embd_dim), Int8QuantScheme::PerRow, true, param->precision);

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

    kv_storage = param->kv_storage;
    kv_cache_layer_id = param->kv_cache_layer_id;
    use_mobilekv = static_cast<bool>(kv_storage) && kv_cache_layer_id >= 0;

    if (use_mobilekv)
    {
        M_Assert(kv_storage->has_layer(static_cast<uint32_t>(kv_cache_layer_id)));
        auto& layer_storage = kv_storage->layer(static_cast<uint32_t>(kv_cache_layer_id));
        auto& k_plane = layer_storage.plane(mobilekv::PlaneKind::K);
        auto& v_plane = layer_storage.plane(mobilekv::PlaneKind::V);
        M_Assert(is_mobilekv_plain_fp32_compatible(k_plane, head_count_kv, embd_dim_head));
        M_Assert(is_mobilekv_plain_fp32_compatible(v_plane, head_count_kv, embd_dim_head));
        cached_len = static_cast<int>(k_plane.stats().seq_length);
    }
    else
    {
        // 预分配 fallback 本地 KV Cache, seq-major layout: [max_seq_len, head_count_kv, embd_dim_head]
        std::vector<int> cache_shape = {max_seq_len, head_count_kv, embd_dim_head};
        k_cache = Mat(cache_shape, DT_32F, 0);
        v_cache = Mat(cache_shape, DT_32F, 0);
        cached_len = 0;
    }
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

static Mat repeat_kv_and_transpose_for_attention(const Mat& x_kv,
                                                 int seq_len,
                                                 int head_count,
                                                 int head_count_kv,
                                                 int embd_dim_head,
                                                 int repeat_kv)
{
    M_Assert(repeat_kv > 1);

    Mat x_kv_hsd({head_count, seq_len, embd_dim_head}, x_kv.type());
    const float* src = reinterpret_cast<const float*>(x_kv.data);  // [S, Hkv, D]
    float* dst = reinterpret_cast<float*>(x_kv_hsd.data);          // [H, S, D]

    const bool parallel_heads = cpu::should_parallelize_1d_loop(
        static_cast<size_t>(head_count),
        static_cast<size_t>(seq_len) * static_cast<size_t>(embd_dim_head),
        1LL << 14,
        1);
#ifdef _OPENMP
#pragma omp parallel for if(parallel_heads)
#endif
    for (int h = 0; h < head_count; ++h)
    {
        const int kv_head = h / repeat_kv;
        float* dst_h = dst + static_cast<size_t>(h) * static_cast<size_t>(seq_len) * static_cast<size_t>(embd_dim_head);
        for (int s = 0; s < seq_len; ++s)
        {
            const float* src_ptr = src + (static_cast<size_t>(s) * static_cast<size_t>(head_count_kv) +
                                          static_cast<size_t>(kv_head)) * static_cast<size_t>(embd_dim_head);
            std::memcpy(dst_h + static_cast<size_t>(s) * static_cast<size_t>(embd_dim_head),
                        src_ptr,
                        static_cast<size_t>(embd_dim_head) * sizeof(float));
        }
    }

    return x_kv_hsd;
}

static Mat repeat_kv_for_attention_legacy(const Mat& x_kv,
                                          int seq_len,
                                          int head_count,
                                          int head_count_kv,
                                          int embd_dim_head,
                                          int repeat_kv)
{
    M_Assert(repeat_kv > 1);
    Mat x_kv_repeated({seq_len, head_count, embd_dim_head}, x_kv.type());
    const float* src = reinterpret_cast<const float*>(x_kv.data);    // [S, Hkv, D]
    float* rep = reinterpret_cast<float*>(x_kv_repeated.data);       // [S, H, D]

    const bool parallel_seq = cpu::should_parallelize_1d_loop(
        static_cast<size_t>(seq_len),
        static_cast<size_t>(head_count) * static_cast<size_t>(embd_dim_head),
        1LL << 14,
        1);
#ifdef _OPENMP
#pragma omp parallel for if(parallel_seq)
#endif
    for (int s = 0; s < seq_len; ++s)
    {
        for (int h = 0; h < head_count; ++h)
        {
            const int kv_head = h / repeat_kv;
            std::memcpy(rep + (static_cast<size_t>(s) * head_count + h) * embd_dim_head,
                        src + (static_cast<size_t>(s) * head_count_kv + kv_head) * embd_dim_head,
                        static_cast<size_t>(embd_dim_head) * sizeof(float));
        }
    }

    return x_kv_repeated;
}

static bool fused_gqa_layout_enabled()
{
    const char* disable_env = std::getenv("MINFER_DISABLE_FUSED_GQA_LAYOUT");
    if (!disable_env || disable_env[0] == '\0')
    {
        return true;
    }
    return disable_env[0] == '0';
}

static bool fused_prefill_sdp_enabled()
{
    const char* disable_env = std::getenv("MINFER_DISABLE_FUSED_PREFILL_SDP");
    if (!disable_env || disable_env[0] == '\0')
    {
        return true;
    }
    return disable_env[0] == '0';
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
    Mat qkv_for_proj;
    if (seq_len == 1)
    {
        // [H, 1, D] and [1, H, D] have the same linear layout when seq_len == 1.
        qkv_for_proj = qkv.reshape({seq_len, head_count * embd_dim_head});
    }
    else
    {
        Mat qkv_t = transposeND(qkv, {1, 0, 2});
        qkv_for_proj = qkv_t.reshape({seq_len, head_count * embd_dim_head});
    }
    project_with_runtime_weight(qkv_for_proj, wout).copyTo(x_out);
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
    if (use_mobilekv)
    {
        auto& layer_storage = kv_storage->layer(static_cast<uint32_t>(kv_cache_layer_id));
        auto& k_plane = layer_storage.plane(mobilekv::PlaneKind::K);
        auto& v_plane = layer_storage.plane(mobilekv::PlaneKind::V);

        M_Assert(layer_storage.append_seq(static_cast<uint32_t>(seq_len)));

        const uint32_t new_len = k_plane.stats().seq_length;
        const uint32_t keep_tokens = std::min<uint32_t>(new_len, static_cast<uint32_t>(seq_len));
        const uint32_t dst_begin = new_len - keep_tokens;
        const int src_begin = seq_len - static_cast<int>(keep_tokens);

        write_plain_fp32_tokens_to_mobilekv(
            k_plane,
            static_cast<uint32_t>(kv_cache_layer_id),
            reinterpret_cast<const float*>(x_k.data),
            seq_len,
            src_begin,
            dst_begin,
            keep_tokens,
            head_count_kv,
            embd_dim_head);
        write_plain_fp32_tokens_to_mobilekv(
            v_plane,
            static_cast<uint32_t>(kv_cache_layer_id),
            reinterpret_cast<const float*>(x_v.data),
            seq_len,
            src_begin,
            dst_begin,
            keep_tokens,
            head_count_kv,
            embd_dim_head);
        cached_len = static_cast<int>(new_len);
    }
    else
    {
        // fallback local cache path
        // x_k & x_v are already [seq_len, head_count_kv, embd_dim_head]
        float* dst_k = reinterpret_cast<float*>(k_cache.data) + static_cast<size_t>(ctx.start_pos) * head_count_kv * embd_dim_head;
        float* dst_v = reinterpret_cast<float*>(v_cache.data) + static_cast<size_t>(ctx.start_pos) * head_count_kv * embd_dim_head;
        const size_t copy_bytes = static_cast<size_t>(seq_len) * head_count_kv * embd_dim_head * sizeof(float);
        memcpy(dst_k, x_k.data, copy_bytes);
        memcpy(dst_v, x_v.data, copy_bytes);

        cached_len = ctx.start_pos + seq_len;
    }

    // Step 5: Prepare attention layout [head, seq, dim].
    Mat x_q_hsd = transposeND(x_q, {1, 0, 2});
    Mat x_k_hsd;
    Mat x_v_hsd;
    const bool use_fused_gqa_layout = fused_gqa_layout_enabled();
    if (repeat_kv <= 1)
    {
        // Keep the original non-GQA path unchanged for numerical stability.
        x_k_hsd = transposeND(x_k, {1, 0, 2});
        x_v_hsd = transposeND(x_v, {1, 0, 2});
    }
    else if (!use_fused_gqa_layout)
    {
        Mat x_k_repeated = repeat_kv_for_attention_legacy(x_k, seq_len, head_count, head_count_kv, embd_dim_head, repeat_kv);
        Mat x_v_repeated = repeat_kv_for_attention_legacy(x_v, seq_len, head_count, head_count_kv, embd_dim_head, repeat_kv);
        x_k_hsd = transposeND(x_k_repeated, {1, 0, 2});
        x_v_hsd = transposeND(x_v_repeated, {1, 0, 2});
    }
    else
    {
        // GQA path: fuse repeat + transpose in one pass to avoid the intermediate repeated [seq, head, dim].
        x_k_hsd = repeat_kv_and_transpose_for_attention(x_k, seq_len, head_count, head_count_kv, embd_dim_head, repeat_kv);
        x_v_hsd = repeat_kv_and_transpose_for_attention(x_v, seq_len, head_count, head_count_kv, embd_dim_head, repeat_kv);
    }

    // Step 6: Attention with causal mask
    Mat qk = gemm(x_q_hsd, x_k_hsd, false, true);
    Mat qk_aligned = runtimePrecision == RuntimePrecision::FP32
        ? qk
        : align_precision_sensitive_input(qk, runtimePrecision);
    const size_t softmax_outer = qk_aligned.total() / (static_cast<size_t>(seq_len) * static_cast<size_t>(seq_len));
    const bool use_fused_prefill_sdp =
        fused_prefill_sdp_enabled() &&
        cpu::should_parallelize_1d_loop(softmax_outer * static_cast<size_t>(seq_len),
                                        static_cast<size_t>(seq_len) * static_cast<size_t>(embd_dim_head),
                                        1LL << 15,
                                        2);
    Mat attn_out;
    if (use_fused_prefill_sdp)
    {
        // Fuse causal softmax + score*V to avoid materializing [H, S, S] softmax probabilities.
        attn_out.create({head_count, seq_len, embd_dim_head}, DT_32F);
        cpu::causal_softmax_weighted_sum_square_xsimd(reinterpret_cast<const float*>(qk_aligned.data),
                                                      reinterpret_cast<const float*>(x_v_hsd.data),
                                                      reinterpret_cast<float*>(attn_out.data),
                                                      softmax_outer,
                                                      seq_len,
                                                      static_cast<size_t>(embd_dim_head),
                                                      1.0f / sqrtf(embd_dim_head));
    }
    else
    {
        cpu::causal_masked_softmax_square_xsimd(reinterpret_cast<const float*>(qk_aligned.data),
                                                reinterpret_cast<float*>(qk_aligned.data),
                                                softmax_outer,
                                                seq_len,
                                                1.0f / sqrtf(embd_dim_head));

        // Step 7: score * V
        attn_out = gemm(qk_aligned, x_v_hsd);
    }

    // Step 8: Transpose back and output linear
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

    Mat k_slice;
    Mat v_slice;
    int total_len = 0;

    // ---------------------------------------------------------------
    // UNIFIED DECODE: Dual-segment direct matmul with Online Softmax (Flash Attention B=1)
    // Works identically for MobileKV and Fallback paths relying on uniform [S, Hkv, D] layout.
    // ---------------------------------------------------------------
    RingSegments kseg, vseg;

    if (use_mobilekv)
    {
        auto& layer_storage = kv_storage->layer(static_cast<uint32_t>(kv_cache_layer_id));
        auto& k_plane = layer_storage.plane(mobilekv::PlaneKind::K);
        auto& v_plane = layer_storage.plane(mobilekv::PlaneKind::V);

        M_Assert(layer_storage.append_seq(1));
        const uint32_t new_len = k_plane.stats().seq_length;
        const uint32_t dst_begin = new_len - 1;

        write_plain_fp32_tokens_to_mobilekv(
            k_plane,
            static_cast<uint32_t>(kv_cache_layer_id),
            reinterpret_cast<const float*>(x_k.data),
            1, 0,
            dst_begin,
            1, head_count_kv, embd_dim_head);
        write_plain_fp32_tokens_to_mobilekv(
            v_plane,
            static_cast<uint32_t>(kv_cache_layer_id),
            reinterpret_cast<const float*>(x_v.data),
            1, 0,
            dst_begin,
            1, head_count_kv, embd_dim_head);

        total_len = static_cast<int>(new_len);
        cached_len = total_len;

        kseg = get_ring_segments(k_plane);
        vseg = get_ring_segments(v_plane);
    }
    else
    {
        // Step 3: Write new K, V to local fallback cache at position cur_pos
        // x_k/x_v shape: [1, head_count_kv, embd_dim_head]
        float* dst_k = reinterpret_cast<float*>(k_cache.data) + static_cast<size_t>(cur_pos) * head_count_kv * embd_dim_head;
        float* dst_v = reinterpret_cast<float*>(v_cache.data) + static_cast<size_t>(cur_pos) * head_count_kv * embd_dim_head;
        const size_t copy_bytes = static_cast<size_t>(head_count_kv) * embd_dim_head * sizeof(float);
        memcpy(dst_k, x_k.data, copy_bytes);
        memcpy(dst_v, x_v.data, copy_bytes);

        cached_len = cur_pos + 1;
        total_len = cached_len;

        // Construct contiguous Segments mimicking ring properties
        kseg.base = reinterpret_cast<const float*>(k_cache.data);
        kseg.write_head = 0;
        kseg.sa = 0;
        kseg.sb = total_len;
        kseg.total = total_len;

        vseg.base = reinterpret_cast<const float*>(v_cache.data);
        vseg.write_head = 0;
        vseg.sa = 0;
        vseg.sb = total_len;
        vseg.total = total_len;
    }

    M_Assert(kseg.total == total_len && vseg.total == total_len);
    M_Assert(kseg.sa == vseg.sa && kseg.sb == vseg.sb);

    const int sa = kseg.sa;
    const int sb = kseg.sb;
    const float scale = 1.0f / sqrtf(static_cast<float>(embd_dim_head));

    // x_q is [1, head_count, embd_dim_head], reshape avoids a copy for seq_len == 1.
    Mat q_t = x_q.reshape({head_count, 1, embd_dim_head});

    // Output of attention: [head_count, 1, embd_dim_head]
    Mat attn_out({head_count, 1, embd_dim_head}, DT_32F, 0.f);

    const float* k_base = reinterpret_cast<const float*>(kseg.base);
    const float* v_base = reinterpret_cast<const float*>(vseg.base);
    float* out_base = reinterpret_cast<float*>(attn_out.data);
    const float* q_base = reinterpret_cast<const float*>(q_t.data);

    const bool parallel_decode_heads = cpu::should_parallelize_1d_loop(
        static_cast<size_t>(head_count),
        static_cast<size_t>(std::max(total_len, 1)) * static_cast<size_t>(embd_dim_head),
        1LL << 14,
        1);
#ifdef _OPENMP
#pragma omp parallel for if(parallel_decode_heads)
#endif
    for (int h = 0; h < head_count; ++h)
    {
        const int h_kv = h / repeat_kv;
        const float* q_ptr = q_base + static_cast<size_t>(h) * embd_dim_head;
        float* out_ptr = out_base + static_cast<size_t>(h) * embd_dim_head;

        float m_curr = -1e30f; // very small float strictly negative
        float l_curr = 0.0f;

        auto process_token = [&](const float* k_ptr, const float* v_ptr) {
            float qk = xsimd_dot_fp32(q_ptr, k_ptr, embd_dim_head) * scale;
            float m_new = std::max(m_curr, qk);
            float exp_diff = std::exp(m_curr - m_new);
            float exp_qk = std::exp(qk - m_new);
            l_curr = l_curr * exp_diff + exp_qk;
            xsimd_fmadd_inplace(out_ptr, exp_diff, exp_qk, v_ptr, embd_dim_head);
            m_curr = m_new;
        };

        // Segment A
        const int sa_end = kseg.write_head + sa;
        for (int t = kseg.write_head; t < sa_end; ++t) {
            const float* k_ptr = k_base + (static_cast<size_t>(t) * head_count_kv + h_kv) * embd_dim_head;
            const float* v_ptr = v_base + (static_cast<size_t>(t) * head_count_kv + h_kv) * embd_dim_head;
            process_token(k_ptr, v_ptr);
        }
        // Segment B
        for (int t = 0; t < sb; ++t) {
            const float* k_ptr = k_base + (static_cast<size_t>(t) * head_count_kv + h_kv) * embd_dim_head;
            const float* v_ptr = v_base + (static_cast<size_t>(t) * head_count_kv + h_kv) * embd_dim_head;
            process_token(k_ptr, v_ptr);
        }

        // Normalize
        for (int d = 0; d < embd_dim_head; ++d) {
            out_ptr[d] /= l_curr;
        }
    }

    Mat out = *output[0];
    project_output_and_add_residual(attn_out, wout, 1, head_count, embd_dim_head, x, out);
}

void AttentionLayer::resetKVCache()
{
    if (use_mobilekv && kv_storage && kv_storage->has_layer(static_cast<uint32_t>(kv_cache_layer_id)))
    {
        kv_storage->layer(static_cast<uint32_t>(kv_cache_layer_id)).clear();
    }
    else
    {
        k_cache.setTo(0.0f);
        v_cache.setTo(0.0f);
    }
    cached_len = 0;
    start_pos = 0;
}

void AttentionLayer::setRuntimePrecision(RuntimePrecision precision)
{
    norm.setPrecision(precision);
    wq.setPrecision(precision);
    wk.setPrecision(precision);
    wv.setPrecision(precision);
    wout.setPrecision(precision);
    Layer::setRuntimePrecision(precision);
}

}
