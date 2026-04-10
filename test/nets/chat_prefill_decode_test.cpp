#include "minfer.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace minfer;

namespace {

std::vector<int> mat_to_token_ids(const Mat& input_ids) {
    MatShape s = input_ids.shape();
    M_Assert(s.size() == 2 && s[0] == 1);
    M_Assert(input_ids.type() == DT_32S);

    std::vector<int> ids(s[1], 0);
    const int* p = reinterpret_cast<const int*>(input_ids.data);
    for (int i = 0; i < s[1]; ++i) {
        ids[i] = p[i];
    }
    return ids;
}

int last_argmax_token(const Mat& logits) {
    std::vector<int> ids = argmax_tokens(
        reinterpret_cast<const float*>(logits.data),
        logits.size[0], logits.size[1], logits.size[2]);
    M_Assert(!ids.empty());
    return ids.back();
}

std::vector<DecodeCandidate> last_topk_tokens(const Mat& logits, int k) {
    M_Assert(k > 0);
    M_Assert(logits.dims == 3);

    const int seq_len = logits.size[1];
    const int vocab_size = logits.size[2];
    const float* row = reinterpret_cast<const float*>(logits.data) + (seq_len - 1) * vocab_size;

    std::vector<DecodeCandidate> topk;
    topk.reserve(vocab_size);
    for (int v = 0; v < vocab_size; ++v) {
        topk.push_back({v, row[v]});
    }

    if (k < static_cast<int>(topk.size())) {
        std::partial_sort(topk.begin(), topk.begin() + k, topk.end(),
                          [](const DecodeCandidate& a, const DecodeCandidate& b) {
                              if (a.logit != b.logit) {
                                  return a.logit > b.logit;
                              }
                              return a.token_id < b.token_id;
                          });
        topk.resize(k);
    } else {
        std::sort(topk.begin(), topk.end(),
                  [](const DecodeCandidate& a, const DecodeCandidate& b) {
                      if (a.logit != b.logit) {
                          return a.logit > b.logit;
                      }
                      return a.token_id < b.token_id;
                  });
    }

    return topk;
}

} // namespace

TEST(Net_TEST, prefill_matches_forward_argmax)
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);

    const std::string root = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    const int num_tests = 4;

    for (int i = 0; i < num_tests; ++i) {
        Mat input_ids = readMatFromNpy(root + "net_input_" + std::to_string(i) + ".npy");
        Mat output_checker = readMatFromNpy(root + "net_output_" + std::to_string(i) + ".npy");

        const std::vector<int> ids = mat_to_token_ids(input_ids);
        net.resetKVCache();
        Mat logits_prefill = net.prefill(ids);

        std::vector<int> tok_prefill = argmax_tokens(
            reinterpret_cast<const float*>(logits_prefill.data),
            logits_prefill.size[0], logits_prefill.size[1], logits_prefill.size[2]);
        std::vector<int> tok_checker = argmax_tokens(
            reinterpret_cast<const float*>(output_checker.data),
            output_checker.size[0], output_checker.size[1], output_checker.size[2]);

        M_Assert(tok_prefill.size() == tok_checker.size());
        for (size_t j = 0; j < tok_prefill.size(); ++j) {
            M_Assert(tok_prefill[j] == tok_checker[j]);
        }
    }
}

TEST(Net_TEST, decode_step_matches_full_forward_next_token)
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);

    const std::string root = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    const int num_tests = 4;

    for (int i = 0; i < num_tests; ++i) {
        Mat input_ids = readMatFromNpy(root + "net_input_" + std::to_string(i) + ".npy");
        std::vector<int> ids = mat_to_token_ids(input_ids);

        net.resetKVCache();
        Mat prefill_logits = net.prefill(ids);
        const int first_gen_token = last_argmax_token(prefill_logits);

        Mat step_logits = net.step(first_gen_token);
        const int step_next_token = last_argmax_token(step_logits);

        // Compare with a full forward pass on [prompt + first_gen_token].
        ids.push_back(first_gen_token);
        Mat ids_ext({1, static_cast<int>(ids.size())}, DT_32S, (void*)ids.data());
        net.setInput(ids_ext);
        net.init();
        Mat full_logits = net.forward();
        const int full_next_token = last_argmax_token(full_logits);

        M_Assert(step_next_token == full_next_token);
    }
}

TEST(Net_TEST, prefill_decode_argmax_matches_full_logits_reference)
{
    Net full_net;
    full_net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);
    Net shortlist_net;
    shortlist_net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);

    const std::string root = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    const int num_tests = 4;

    for (int i = 0; i < num_tests; ++i) {
        Mat input_ids = readMatFromNpy(root + "net_input_" + std::to_string(i) + ".npy");
        std::vector<int> ids = mat_to_token_ids(input_ids);

        full_net.resetKVCache();
        shortlist_net.resetKVCache();
        Mat full_logits = full_net.prefill(ids);
        DecodeResult decode_result = shortlist_net.prefillDecode(ids, DecodeOutputMode::ArgMax);

        EXPECT_EQ(decode_result.mode, DecodeOutputMode::ArgMax);
        EXPECT_TRUE(decode_result.logits.empty());
        EXPECT_EQ(decode_result.token_id, last_argmax_token(full_logits));
        EXPECT_TRUE(decode_result.top_k.empty());
    }
}

TEST(Net_TEST, step_decode_topk_matches_full_logits_reference)
{
    Net full_net;
    full_net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);
    Net shortlist_net;
    shortlist_net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);

    const std::string root = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    const int num_tests = 4;
    const int k = 5;

    for (int i = 0; i < num_tests; ++i) {
        Mat input_ids = readMatFromNpy(root + "net_input_" + std::to_string(i) + ".npy");
        std::vector<int> ids = mat_to_token_ids(input_ids);

        full_net.resetKVCache();
        shortlist_net.resetKVCache();
        Mat prefill_logits = full_net.prefill(ids);
        shortlist_net.prefill(ids);
        const int first_gen_token = last_argmax_token(prefill_logits);

        Mat full_logits = full_net.step(first_gen_token);
        DecodeResult decode_result = shortlist_net.stepDecode(first_gen_token, DecodeOutputMode::TopK, k);
        std::vector<DecodeCandidate> ref_topk = last_topk_tokens(full_logits, k);

        ASSERT_EQ(decode_result.mode, DecodeOutputMode::TopK);
        ASSERT_TRUE(decode_result.logits.empty());
        ASSERT_EQ(decode_result.top_k.size(), ref_topk.size());
        ASSERT_EQ(decode_result.token_id, ref_topk.front().token_id);

        for (size_t j = 0; j < ref_topk.size(); ++j) {
            EXPECT_EQ(decode_result.top_k[j].token_id, ref_topk[j].token_id);
            EXPECT_FLOAT_EQ(decode_result.top_k[j].logit, ref_topk[j].logit);
        }
    }
}

TEST(Net_TEST, shortlist_argmax_generation_loop_matches_full_logits_loop)
{
    Net full_net;
    full_net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);
    Net shortlist_net;
    shortlist_net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);

    const std::string root = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    const int num_tests = 4;
    const int gen_steps = 8;

    for (int i = 0; i < num_tests; ++i) {
        Mat input_ids = readMatFromNpy(root + "net_input_" + std::to_string(i) + ".npy");
        std::vector<int> ids = mat_to_token_ids(input_ids);

        full_net.resetKVCache();
        shortlist_net.resetKVCache();

        std::vector<int> full_tokens;
        std::vector<int> shortlist_tokens;
        full_tokens.reserve(gen_steps);
        shortlist_tokens.reserve(gen_steps);

        Mat full_prefill_logits = full_net.prefill(ids);
        DecodeResult shortlist_prefill = shortlist_net.prefillDecode(ids, DecodeOutputMode::ArgMax);

        int full_next = last_argmax_token(full_prefill_logits);
        int shortlist_next = shortlist_prefill.token_id;

        for (int step = 0; step < gen_steps; ++step) {
            full_tokens.push_back(full_next);
            shortlist_tokens.push_back(shortlist_next);

            Mat full_step_logits = full_net.step(full_next);
            DecodeResult shortlist_step = shortlist_net.stepDecode(shortlist_next, DecodeOutputMode::ArgMax);

            full_next = last_argmax_token(full_step_logits);
            shortlist_next = shortlist_step.token_id;
        }

        EXPECT_EQ(shortlist_tokens, full_tokens);
    }
}
