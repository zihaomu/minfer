#include "minfer.h"
#include "gtest/gtest.h"

#include "../src/backend/cpu/layer/attention_layer.h"
#include "../src/backend/cpu/layer/embeding_layer.h"
#include "../src/backend/cpu/layer/feed_forward.h"
#include "../src/backend/cpu/layer/linear_layer.h"
#include "../src/backend/cpu/layer/rms_norm_layer.h"

#include <algorithm>
#include <cmath>

using namespace minfer;

namespace {

std::string layer_data_path(const std::string& filename)
{
    return std::string(M_ROOT_PATH) + "/test/core/test_data/data/" + filename;
}

void fill_trig(Mat& mat, float scale = 1.0f, float bias = 0.0f)
{
    float* data = reinterpret_cast<float*>(mat.data);
    for (size_t i = 0; i < mat.total(); ++i)
    {
        data[i] = std::sin(static_cast<float>(i) * 0.17f) * scale +
                  std::cos(static_cast<float>(i) * 0.07f) * scale * 0.5f +
                  bias;
    }
}

void expect_close_stats(const Mat& out, const Mat& ref, float mean_abs_tol, float max_abs_tol, const char* name)
{
    ASSERT_EQ(out.shape(), ref.shape()) << name;
    const float* out_data = reinterpret_cast<const float*>(out.data);
    const float* ref_data = reinterpret_cast<const float*>(ref.data);

    double mean_abs = 0.0;
    double max_abs = 0.0;
    for (size_t i = 0; i < out.total(); ++i)
    {
        const double diff = std::abs(static_cast<double>(out_data[i]) - static_cast<double>(ref_data[i]));
        mean_abs += diff;
        max_abs = std::max(max_abs, diff);
    }
    mean_abs /= std::max<size_t>(1, out.total());

    EXPECT_LE(mean_abs, mean_abs_tol) << name << " mean_abs=" << mean_abs;
    EXPECT_LE(max_abs, max_abs_tol) << name << " max_abs=" << max_abs;
}

Mat make_ids(const std::vector<int>& ids)
{
    return Mat({1, static_cast<int>(ids.size())}, DT_32S, const_cast<int*>(ids.data())).clone();
}

template <typename LayerPtr>
Mat run_layer(const LayerPtr& layer, Mat& input, const MatShape& output_shape)
{
    Mat output(output_shape, DT_32F);
    std::vector<Mat*> inputs = {&input};
    std::vector<Mat*> outputs = {&output};
    layer->init(inputs, outputs);
    layer->forward(inputs, outputs);
    return output;
}

}  // namespace

TEST(Layer_TEST, quantized_linear_runtime_precision_matches_fp32)
{
    Mat input({1, 3, 8}, DT_32F);
    Mat weight({6, 8}, DT_32F);
    Mat bias({6}, DT_32F);
    fill_trig(input, 0.8f);
    fill_trig(weight, 0.5f);
    fill_trig(bias, 0.2f);

    auto make_layer = [&](RuntimePrecision precision) {
        auto params = std::shared_ptr<LinearLayerParams>(new LinearLayerParams({0}, {1}, 8, 6, weight, bias));
        params->precision = precision;
        return LinearLayer::create(params);
    };

    Mat fp32_out = run_layer(make_layer(RuntimePrecision::FP32), input, {1, 3, 6});
    Mat fp16_out = run_layer(make_layer(RuntimePrecision::FP16), input, {1, 3, 6});
    expect_close_stats(fp16_out, fp32_out, 3e-2f, 1.2e-1f, "linear_fp16");

    Mat int8_out = run_layer(make_layer(RuntimePrecision::INT8), input, {1, 3, 6});
    expect_close_stats(int8_out, fp32_out, 1.2e-1f, 5e-1f, "linear_int8");
}

TEST(Layer_TEST, quantized_linear_matches_python_precision_references)
{
    Mat input = readMatFromNpy(layer_data_path("linear_precision_input.npy"));
    Mat weight = readMatFromNpy(layer_data_path("linear_precision_weight.npy"));
    Mat bias = readMatFromNpy(layer_data_path("linear_precision_bias.npy"));
    Mat ref_fp32 = readMatFromNpy(layer_data_path("linear_precision_fp32_o.npy"));
    Mat ref_fp16 = readMatFromNpy(layer_data_path("linear_precision_fp16_o.npy"));
    Mat ref_int8 = readMatFromNpy(layer_data_path("linear_precision_int8_o.npy"));

    auto make_layer = [&](RuntimePrecision precision) {
        auto params = std::shared_ptr<LinearLayerParams>(new LinearLayerParams({0}, {1}, 8, 6, weight, bias));
        params->precision = precision;
        return LinearLayer::create(params);
    };

    Mat fp32_out = run_layer(make_layer(RuntimePrecision::FP32), input, {1, 3, 6});
    expect_close_stats(fp32_out, ref_fp32, 1e-6f, 1e-6f, "linear_python_fp32");

    Mat fp16_out = run_layer(make_layer(RuntimePrecision::FP16), input, {1, 3, 6});
    expect_close_stats(fp16_out, ref_fp16, 2e-3f, 1e-2f, "linear_python_fp16");

    Mat int8_out = run_layer(make_layer(RuntimePrecision::INT8), input, {1, 3, 6});
    expect_close_stats(int8_out, ref_int8, 2e-3f, 1e-2f, "linear_python_int8");
}

TEST(Layer_TEST, quantized_embedding_rmsnorm_ffn_attention_match_fp32)
{
    const int vocab = 16;
    const int embd = 8;
    const int seq_len = 4;
    const int heads = 2;
    const int ffn_dim = 16;

    Mat ids = make_ids({1, 3, 7, 2});
    Mat emb_w({vocab, embd}, DT_32F);
    Mat norm_w({embd}, DT_32F);
    Mat wq({embd, embd}, DT_32F);
    Mat wk({embd, embd}, DT_32F);
    Mat wv({embd, embd}, DT_32F);
    Mat wout({embd, embd}, DT_32F);
    Mat gate({ffn_dim, embd}, DT_32F);
    Mat up({ffn_dim, embd}, DT_32F);
    Mat down({embd, ffn_dim}, DT_32F);

    fill_trig(emb_w, 0.4f);
    fill_trig(norm_w, 0.2f, 1.0f);
    fill_trig(wq, 0.3f);
    fill_trig(wk, 0.3f);
    fill_trig(wv, 0.3f);
    fill_trig(wout, 0.3f);
    fill_trig(gate, 0.25f);
    fill_trig(up, 0.2f);
    fill_trig(down, 0.2f);

    auto run_stack = [&](RuntimePrecision precision) {
        auto embedding_params = std::shared_ptr<EmbeddingLayerParams>(
            new EmbeddingLayerParams({0}, {1}, vocab, embd, emb_w));
        auto attention_params = std::shared_ptr<AttentionLayerParams>(
            new AttentionLayerParams({1}, {2}, 8, embd, heads, heads, 1e-6f, norm_w, wq, wk, wv, wout));
        auto ffn_params = std::shared_ptr<FeedForwardLayerParams>(
            new FeedForwardLayerParams({2}, {3}, ActivateType::SILU, embd, ffn_dim, 1e-6f, norm_w, gate, up, down));
        auto rms_params = std::shared_ptr<RMSNormLayerParams>(
            new RMSNormLayerParams({3}, {4}, embd, 1e-6f, norm_w));

        embedding_params->precision = precision;
        attention_params->precision = precision;
        ffn_params->precision = precision;
        rms_params->precision = precision;

        auto embedding = EmbeddingLayer::create(embedding_params);
        auto attention = AttentionLayer::create(attention_params);
        auto ffn = FeedForwardLayer::create(ffn_params);
        auto rms = RMSNormLayer::create(rms_params);

        Mat emb_out = run_layer(embedding, ids, {1, seq_len, embd});
        attention->resetKVCache();
        Mat attn_out = run_layer(attention, emb_out, {1, seq_len, embd});
        Mat ffn_out = run_layer(ffn, attn_out, {1, seq_len, embd});
        return run_layer(rms, ffn_out, {1, seq_len, embd});
    };

    Mat fp32_out = run_stack(RuntimePrecision::FP32);
    Mat fp16_out = run_stack(RuntimePrecision::FP16);
    expect_close_stats(fp16_out, fp32_out, 8e-2f, 4e-1f, "stack_fp16");

    Mat int8_out = run_stack(RuntimePrecision::INT8);
    expect_close_stats(int8_out, fp32_out, 2.5e-1f, 1.5f, "stack_int8");
}
