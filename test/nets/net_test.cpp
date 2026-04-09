//
// Created by mzh on 2024/4/1.
//

#include "minfer.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace minfer;

namespace
{

template <typename T, typename = void>
struct has_set_precision : std::false_type
{
};

template <typename T>
struct has_set_precision<T, std::void_t<decltype(std::declval<T&>().setPrecision(RuntimePrecision::FP16))>>
    : std::true_type
{
};

template <typename T, typename = void>
struct has_set_runtime_precision : std::false_type
{
};

template <typename T>
struct has_set_runtime_precision<T, std::void_t<decltype(std::declval<T&>().setRuntimePrecision(RuntimePrecision::FP16))>>
    : std::true_type
{
};

static_assert(!has_set_precision<Net>::value, "Net::setPrecision should not be exposed in public API");
static_assert(!has_set_runtime_precision<Net>::value, "Net::setRuntimePrecision should not be exposed in public API");

} // namespace

// TODO add test element equal check. compare two mat, or compare mat and scalar.
TEST(Net_TEST, simple_net_test)
{
    float a = 20.f;
    int intValue = *reinterpret_cast<int*>(&a);

    std::cout << "Float value: " << a << std::endl;
    std::cout << "Reinterpreted int value: " << intValue << std::endl;


    std::vector<std::shared_ptr<LayerParams> > layers =
            {
                    std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1})),
                    std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {2}, {3})),
                    std::shared_ptr<LayerParams>(new LayerParams(LayerType::Add, {1,3}, {4})),
                    std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {4}, {5}))
            };
    Net net_v0;
    net_v0.createNet(layers);

    std::shared_ptr<LayerParams> input0 = std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1}));
    std::shared_ptr<LayerParams> input1 = std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {2}, {3}));
    std::shared_ptr<LayerParams> add = std::shared_ptr<LayerParams>(new LayerParams(LayerType::Add, {1,3}, {4}));
    std::shared_ptr<LayerParams> out = std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {4}, {5}));

    Net net_v1;
    net_v1.createLayer(input0);
    net_v1.createLayer(input1);
    net_v1.createLayer(add);
    net_v1.createLayer(out);

    float f20 = 20.f;
    float f30 = 30.f;
    float f50 = 50.f;
    Mat inpM1 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f20));
    Mat inpM2 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f30));
    Mat outM  = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f50));

    inpM1.print();
    inpM2.print();

    net_v0.setInput(inpM1, 0);
    net_v0.setInput(inpM2, 2);

    net_v0.init();
    Mat outMat_0 = net_v0.forward();

    net_v1.setInput(inpM1, 0);
    net_v1.setInput(inpM2, 2);

    net_v1.init();
    Mat outMat_1 = net_v1.forward();

    outMat_0.print();
    outMat_1.print();
}

TEST(Net_TEST, tokenizer)
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP16);
    EXPECT_EQ(net.getPrecision(), RuntimePrecision::FP16);

    std::vector<int> ids_ground_truth = {1, 22557, 1526, 28808, 523, 28713, 28767};

    std::string text = "Hello world! <s>";
    std::vector<int> ids;

    // tokenizer
    net.encode(text, ids);

    for (int i = 0; i < ids_ground_truth.size(); i++)
    {
        M_Assert(ids[i] == ids_ground_truth[i]);
    }

    std::cout << "Token IDs: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;

    std::string out_text;
    net.decode(ids, out_text);

    std::cout << "Decoded Text: " << out_text << std::endl;

}

TEST(Net_TEST, net_tiny_llama)
{
    std::cout << "print test on net_tiny_llama" << std::endl;
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf");
    EXPECT_EQ(net.getPrecision(), RuntimePrecision::FP32);

    std::string ROOT_path = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";

    int num_tests = 4;
    for (int i = 0; i < num_tests; i++) {
        std::string input_path = ROOT_path + "net_input_" + std::to_string(i) + ".npy";
        std::string output_path = ROOT_path + "net_output_" + std::to_string(i) + ".npy";

        Mat input_ids = readMatFromNpy(input_path);
        Mat output_checker = readMatFromNpy(output_path);

        net.setInput(input_ids);
        net.init();

        Mat output = net.forward();


        std::vector<int> token_ids_checker = argmax_tokens(reinterpret_cast<const float*>(output_checker.data), output_checker.size[0], output_checker.size[1], output_checker.size[2]);
        std::vector<int> token_ids = argmax_tokens(reinterpret_cast<const float*>(output.data), output.size[0], output.size[1], output.size[2]);

        for (int j = 0; j < token_ids.size(); j++)
        {
            M_Assert(token_ids[j] == token_ids_checker[j]);
        }
        
        std::string out_text, checker_text;
        net.decode(token_ids, out_text);
        net.decode(token_ids_checker, checker_text);
        
        std::cout << "Decoded Text: " << out_text << std::endl;
        std::cout << "Checker Text: " << checker_text << std::endl;

        std::cout << "Running forward pass for prompt " << i << std::endl;

        // dump internal tensors if accessible, otherwise just print output.
        // To directly check Embedding, we would need to access net.layers_[1]->output[0] etc.
        // Assuming we can access layers_ if public, else skip
        // checking the first 10 float values of the final output explicitly.

        std::cout << "Engine output shape: ";
        for (int d = 0; d < output.dims; ++d) std::cout << output.size[d] << " ";
        std::cout << "\nChecker output shape: ";
        for (int d = 0; d < output_checker.dims; ++d) std::cout << output_checker.size[d] << " ";
        std::cout << std::endl;
        
        // Since we explicitly save [1, seq_len, vocab_size] from python, the shapes match exactly
        double mean_l1 = norm(output, output_checker, NORM_L1) / output.total();
        double rel_l2_a  = norm(output, output_checker, NORM_L2);
        double rel_l2_b = norm(output_checker, NORM_L2) + 1e-12;
        double rel_l2 = rel_l2_a / rel_l2_b;
        double max_err = norm(output, output_checker, NORM_INF);

        std::cout<<"output_checker"<<std::endl;
        output_checker.print(10);
        std::cout<<"output"<<std::endl;
        output.print(10);
        // M_Assert(mean_l1 < 1);
        // M_Assert(rel_l2_a  < 1e-5);
        // M_Assert(rel_l2_b  < 1e-5);
        // M_Assert(max_err < 2);

        std::cout << "Prompt " << i << " -> mean L1 = " << mean_l1
                  << ", relative L2 = " << rel_l2
                  << ", max abs = " << max_err << std::endl;

        if (i == 0) {
            const float* outs = reinterpret_cast<const float*>(output.data);
            int vocab_size = output.size[2]; // Assuming output shape is [batch, seq_len, vocab_size]
            std::cout << "DEBUG: C++ Logits Token 0 First 10: ";
            for (int d = 0; d < 10; d++) {
                std::cout << outs[d] << " ";
            }
            std::cout << std::endl;
            
            std::cout << "DEBUG: C++ Logits Token 1 First 10: ";
            for (int d = 0; d < 10; d++) {
                std::cout << outs[1 * vocab_size + d] << " ";
            }
            std::cout << std::endl;
        }
    }
}

TEST(Net_TEST, runtime_precision_matches_fp32_on_synthetic_transformer)
{
    const int vocab = 16;
    const int embd = 8;
    const int seq_len = 4;
    const int heads = 2;
    const int ffn_dim = 16;

    auto make_mat = [](const std::vector<int>& shape, float scale = 1.0f, float bias = 0.0f) {
        Mat mat(shape, DT_32F);
        float* data = reinterpret_cast<float*>(mat.data);
        for (size_t i = 0; i < mat.total(); ++i)
        {
            data[i] = std::sin(static_cast<float>(i) * 0.17f) * scale +
                      std::cos(static_cast<float>(i) * 0.05f) * scale * 0.5f +
                      bias;
        }
        return mat;
    };

    std::vector<std::shared_ptr<LayerParams>> layers = {
        std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1})),
        std::shared_ptr<LayerParams>(new EmbeddingLayerParams({1}, {2}, vocab, embd, make_mat({vocab, embd}, 0.4f))),
        std::shared_ptr<LayerParams>(new AttentionLayerParams({2}, {3}, 8, embd, heads, heads, 1e-6f,
            make_mat({embd}, 0.2f, 1.0f), make_mat({embd, embd}, 0.25f), make_mat({embd, embd}, 0.25f),
            make_mat({embd, embd}, 0.25f), make_mat({embd, embd}, 0.25f))),
        std::shared_ptr<LayerParams>(new FeedForwardLayerParams({3}, {4}, ActivateType::SILU, embd, ffn_dim, 1e-6f,
            make_mat({embd}, 0.2f, 1.0f), make_mat({ffn_dim, embd}, 0.2f), make_mat({ffn_dim, embd}, 0.2f),
            make_mat({embd, ffn_dim}, 0.2f))),
        std::shared_ptr<LayerParams>(new LinearLayerParams({4}, {5}, embd, vocab, make_mat({vocab, embd}, 0.2f), make_mat({vocab}, 0.05f))),
        std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {5}, {6})),
    };

    std::vector<int> ids_data = {1, 3, 7, 2};
    Mat input_ids({1, seq_len}, DT_32S, ids_data.data());

    auto run_net = [&](RuntimePrecision precision) {
        Net net;
        net.createNet(layers, precision);
        net.setInput(input_ids);
        net.init();
        return net.forward();
    };

    Mat fp32_out = run_net(RuntimePrecision::FP32);
    Mat fp16_out = run_net(RuntimePrecision::FP16);
    Mat int8_out = run_net(RuntimePrecision::INT8);

    auto expect_close = [](const Mat& out, const Mat& ref, float mean_tol, float max_tol, const char* name) {
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
        EXPECT_LE(mean_abs, mean_tol) << name << " mean_abs=" << mean_abs;
        EXPECT_LE(max_abs, max_tol) << name << " max_abs=" << max_abs;
    };

    expect_close(fp16_out, fp32_out, 2.0e-1f, 1.2f, "net_fp16");
    expect_close(int8_out, fp32_out, 4.0e-1f, 2.5f, "net_int8");
}

TEST(Net_TEST, create_net_default_precision_matches_explicit_fp32)
{
    const int vocab = 16;
    const int embd = 8;
    const int seq_len = 4;
    const int heads = 2;
    const int ffn_dim = 16;

    auto make_mat = [](const std::vector<int>& shape, float scale = 1.0f, float bias = 0.0f) {
        Mat mat(shape, DT_32F);
        float* data = reinterpret_cast<float*>(mat.data);
        for (size_t i = 0; i < mat.total(); ++i)
        {
            data[i] = std::sin(static_cast<float>(i) * 0.11f) * scale +
                      std::cos(static_cast<float>(i) * 0.03f) * scale * 0.5f +
                      bias;
        }
        return mat;
    };

    std::vector<std::shared_ptr<LayerParams>> layers = {
        std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1})),
        std::shared_ptr<LayerParams>(new EmbeddingLayerParams({1}, {2}, vocab, embd, make_mat({vocab, embd}, 0.4f))),
        std::shared_ptr<LayerParams>(new AttentionLayerParams({2}, {3}, 8, embd, heads, heads, 1e-6f,
            make_mat({embd}, 0.2f, 1.0f), make_mat({embd, embd}, 0.25f), make_mat({embd, embd}, 0.25f),
            make_mat({embd, embd}, 0.25f), make_mat({embd, embd}, 0.25f))),
        std::shared_ptr<LayerParams>(new FeedForwardLayerParams({3}, {4}, ActivateType::SILU, embd, ffn_dim, 1e-6f,
            make_mat({embd}, 0.2f, 1.0f), make_mat({ffn_dim, embd}, 0.2f), make_mat({ffn_dim, embd}, 0.2f),
            make_mat({embd, ffn_dim}, 0.2f))),
        std::shared_ptr<LayerParams>(new LinearLayerParams({4}, {5}, embd, vocab, make_mat({vocab, embd}, 0.2f), make_mat({vocab}, 0.05f))),
        std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {5}, {6})),
    };

    std::vector<int> ids_data = {1, 3, 7, 2};
    Mat input_ids({1, seq_len}, DT_32S, ids_data.data());

    auto run_with_create_overload = [&](RuntimePrecision precision) {
        Net net;
        net.createNet(layers, precision);
        EXPECT_EQ(net.getPrecision(), precision);
        net.setInput(input_ids);
        net.init();
        return net.forward();
    };

    auto run_with_default_create = [&]() {
        Net net;
        net.createNet(layers);
        EXPECT_EQ(net.getPrecision(), RuntimePrecision::FP32);
        net.setInput(input_ids);
        net.init();
        return net.forward();
    };

    Mat fp32_explicit = run_with_create_overload(RuntimePrecision::FP32);
    Mat fp32_default = run_with_default_create();

    auto expect_close = [](const Mat& out, const Mat& ref, float mean_tol, float max_tol, const char* name) {
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
        EXPECT_LE(mean_abs, mean_tol) << name << " mean_abs=" << mean_abs;
        EXPECT_LE(max_abs, max_tol) << name << " max_abs=" << max_abs;
    };

    expect_close(fp32_default, fp32_explicit, 1e-6f, 1e-6f, "net_default_precision_fp32_matches_explicit");
}
