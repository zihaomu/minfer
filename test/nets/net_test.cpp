//
// Created by mzh on 2024/4/1.
//

#include "minfer.h"
#include "gtest/gtest.h"
#include "sentencepiece_processor.h"

using namespace minfer;

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

// TEST(Net_TEST, multi_head_transform_test)
// {
//     // 参数目前为空
//     int head_count;
//     int head_count_kv;
//     Mat _q;
//     Mat _k;
//     Mat _v;
//     Mat _out;
//
//     // 需要增加参数，减少layer和layer name的要求。
//     std::vector<std::shared_ptr<LayerParams> > layers = {
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1})),
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::RMSNorm, {1}, {2})),
//             std::shared_ptr<LayerParams>(new AttentionLayerParams(LayerType::Attention,{2}, {3}, head_count,
//                                                                   head_count_kv, _q, _k, _v, _out)),
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::Add, {1,3}, {4})),
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::RMSNorm, {4}, {5})),
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::FFN, {5}, {6})),
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::Add, {4,6}, {7})),
//             std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {7}, {8})),
//     };
//
//     Net nets;
//     nets.createNet(layers);
//
//     float f20 = 20.f;
//     float f30 = 30.f;
//     float f50 = 50.f;
//     Mat inpM1 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f20));
//     Mat inpM2 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f30));
//     Mat outM  = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f50));
//
//     inpM1.print();
//     inpM2.print();
//
//     nets.setInput(inpM1, 0);
//     nets.setInput(inpM2, 2);
//
//     nets.init();
//     Mat outMat = nets.forward();
//
//     outM.print();
// }

static std::vector<int> argmax_tokens(const float* logits, int batch, int seq_len, int vocab_size) {
    std::vector<int> token_ids(seq_len, -1);

    // 只取 batch=0 的情况
    for (int t = 0; t < seq_len; t++) {
        const float* row = logits + t * vocab_size;
        int best_id = 0;
        float best_val = -std::numeric_limits<float>::infinity();

        for (int v = 0; v < vocab_size; v++) {
            float val = row[v];
            if (val > best_val) {
                best_val = val;
                best_id = v;
            }
        }
        token_ids[t] = best_id;
    }
    return token_ids;
}

static std::string SafeDecodeTokens(
    sentencepiece::SentencePieceProcessor& sp,
    const std::vector<int>& tokens
) {
    int vocab_size = sp.GetPieceSize();
    std::string decoded;

    for (int t : tokens) {
        if (t >= 0 && t < vocab_size) {
            // 合法 token，用 SentencePiece 解码
            decoded += sp.IdToPiece(t);
        } else if (t >= vocab_size) {
            // 超出 vocab 的 token，标记为 <unused>
            decoded += "<unused_" + std::to_string(t) + ">";
        } else {
            // 不合法 token，负值
            decoded += "<invalid_" + std::to_string(t) + ">";
        }
    }

    return decoded;
}

TEST(Net_TEST, net_tiny_llama)
{
    std::cout<<"print test on net_tiny_llama"<<std::endl;
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf");

    sentencepiece::SentencePieceProcessor sp;
    auto status = sp.Load(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16_tokenizer.model");
    if (!status.ok()) {
        std::cerr << "Failed to load Lite-Oute-1-65M-FP16_tokenizer.model: " << status.ToString() << std::endl;
    }
    // 获取 vocab 大小
    int vocab_size = sp.GetPieceSize();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    std::string text = "Hello, how are you?";
    std::vector<int> ids;
    sp.Encode(text, &ids);

    std::cout << "Token IDs: ";
    for (int id : ids) std::cout << id << " ";
    std::cout << std::endl;

    // convert tokens id to Mat
    std::vector<int> mat_shape = {1, (int)ids.size()};

    Mat input = Mat(mat_shape, DT_32S, ids.data());

    net.setInput(input);
    net.init();

    Mat output = net.forward();
    net.forward(output);

    output.print(10);

    std::vector<int> out_idx = argmax_tokens((float*)output.data, 1, output.size[1], output.size[2]);

    std::cout << "Out Token IDs: ";
    for (int id : out_idx) std::cout << id << " ";
    std::cout << std::endl;

    std::string out_text;
    out_text = SafeDecodeTokens(sp, out_idx);

    for (int t : out_idx) {
        if (t < 0 || t >= sp.GetPieceSize()) {
            std::cerr << "Invalid token ID: " << t << std::endl;
        }
    }

    std::cout << "Output Text: " << out_text << std::endl;

    // TODO add the forward type.
}
