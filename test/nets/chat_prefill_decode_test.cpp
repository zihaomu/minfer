#include "minfer.h"
#include "gtest/gtest.h"

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
