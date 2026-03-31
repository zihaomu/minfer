#include "minfer.h"
#include "gtest/gtest.h"
#include "mobilekv/kv_cache.h"
#include "../src/backend/cpu/layer/attention_layer.h"

#include <cstring>
#include <fstream>
#include <memory>
#include <string>

using namespace minfer;

namespace {

std::string write_mobilekv_cfg(int num_heads_kv, int head_dim, int max_seq_len) {
    const std::string cfg_path = "/tmp/minfer_attention_mobilekv_test.cfg";

    std::ofstream cfg(cfg_path, std::ios::out | std::ios::trunc);
    cfg << "model num_heads=" << num_heads_kv << " head_dim=" << head_dim << "\n";
    cfg << "storage default_alignment=64 thread_safe=false default_max_seq_capacity=" << max_seq_len << "\n";
    cfg << "defaults k_type=fp32 v_type=fp32 initial=" << max_seq_len << " max=" << max_seq_len << "\n";
    cfg << "group 0-0\n";
    cfg.close();

    return cfg_path;
}

Mat slice_first_tokens(const Mat& src, int seq_len, int embd_dim) {
    M_Assert(src.type() == DT_32F);
    Mat out({1, seq_len, embd_dim}, DT_32F);
    const size_t elems = static_cast<size_t>(seq_len) * static_cast<size_t>(embd_dim);
    std::memcpy(out.data, src.data, elems * sizeof(float));
    return out;
}

}  // namespace

TEST(Layer_TEST, attention_mobilekv_prefill_decode_matches_local_cache)
{
    const std::string root = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    Mat input_full = readMatFromNpy(root + "atten_input.npy");
    Mat wq = readMatFromNpy(root + "atten_params_0.npy");
    Mat wk = readMatFromNpy(root + "atten_params_1.npy");
    Mat wv = readMatFromNpy(root + "atten_params_2.npy");
    Mat wout = readMatFromNpy(root + "atten_params_3.npy");
    Mat wrms = readMatFromNpy(root + "atten_rms_params.npy");

    const float rms_eps = 1e-6f;
    const int embd_dim = 128;
    const int head_count = 8;
    const int head_count_kv = 8;
    const int max_seq_len = 256;
    const int prefill_len = 64;
    const int head_dim = embd_dim / head_count;

    Mat input_prefill = slice_first_tokens(input_full, prefill_len, embd_dim);

    auto local_param = std::shared_ptr<AttentionLayerParams>(new AttentionLayerParams(
        {0}, {1}, max_seq_len, embd_dim, head_count, head_count_kv, rms_eps,
        wrms, wq, wk, wv, wout));

    auto mobile_param = std::shared_ptr<AttentionLayerParams>(new AttentionLayerParams(
        {0}, {1}, max_seq_len, embd_dim, head_count, head_count_kv, rms_eps,
        wrms, wq, wk, wv, wout));

    const std::string cfg_path = write_mobilekv_cfg(head_count_kv, head_dim, max_seq_len);
    std::string cfg_error;
    auto storage_unique = mobilekv::create_storage_from_config_file(cfg_path, &cfg_error);
    ASSERT_TRUE(storage_unique != nullptr) << cfg_error;

    auto shared_storage = std::shared_ptr<mobilekv::KVCacheStorage>(std::move(storage_unique));
    mobile_param->kv_storage = shared_storage;
    mobile_param->kv_cache_layer_id = 0;

    auto layer_local = AttentionLayer::create(local_param);
    auto layer_mobile = AttentionLayer::create(mobile_param);

    Mat prefill_out_local({1, prefill_len, embd_dim}, DT_32F);
    Mat prefill_out_mobile({1, prefill_len, embd_dim}, DT_32F);

    std::vector<Mat*> prefill_inputs = {&input_prefill};
    std::vector<Mat*> prefill_outs_local = {&prefill_out_local};
    std::vector<Mat*> prefill_outs_mobile = {&prefill_out_mobile};

    InferenceContext prefill_ctx;
    prefill_ctx.phase = InferPhase::Prefill;
    prefill_ctx.start_pos = 0;
    prefill_ctx.seq_len = prefill_len;

    layer_local->forward(prefill_inputs, prefill_outs_local, prefill_ctx);
    layer_mobile->forward(prefill_inputs, prefill_outs_mobile, prefill_ctx);

    const double prefill_rel_l2 = norm(prefill_out_local, prefill_out_mobile, NORM_L2) /
                                  (norm(prefill_out_local, NORM_L2) + 1e-12);
    const double prefill_max_err = norm(prefill_out_local, prefill_out_mobile, NORM_INF);

    M_Assert(prefill_rel_l2 < 1e-6);
    M_Assert(prefill_max_err < 1e-4);

    Mat decode_input({1, 1, embd_dim}, DT_32F);
    std::memcpy(decode_input.data,
                reinterpret_cast<const float*>(input_prefill.data) + (prefill_len - 1) * embd_dim,
                static_cast<size_t>(embd_dim) * sizeof(float));

    Mat decode_out_local({1, 1, embd_dim}, DT_32F);
    Mat decode_out_mobile({1, 1, embd_dim}, DT_32F);
    std::vector<Mat*> decode_inputs = {&decode_input};
    std::vector<Mat*> decode_outs_local = {&decode_out_local};
    std::vector<Mat*> decode_outs_mobile = {&decode_out_mobile};

    InferenceContext decode_ctx;
    decode_ctx.phase = InferPhase::Decode;
    decode_ctx.start_pos = prefill_len;
    decode_ctx.seq_len = 1;

    layer_local->forward(decode_inputs, decode_outs_local, decode_ctx);
    layer_mobile->forward(decode_inputs, decode_outs_mobile, decode_ctx);

    const double decode_rel_l2 = norm(decode_out_local, decode_out_mobile, NORM_L2) /
                                 (norm(decode_out_local, NORM_L2) + 1e-12);
    const double decode_max_err = norm(decode_out_local, decode_out_mobile, NORM_INF);

    M_Assert(decode_rel_l2 < 1e-6);
    M_Assert(decode_max_err < 1e-4);
}
