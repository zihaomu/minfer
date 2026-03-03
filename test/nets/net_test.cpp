//
// Created by mzh on 2024/4/1.
//

#include "minfer.h"
#include "gtest/gtest.h"

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

TEST(Net_TEST, tokenizer)
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf");

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
