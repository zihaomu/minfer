//
// Created by mzh on 2024/10/29.
//

#include "minfer.h"
#include "gtest/gtest.h"
#include "../src/backend/cpu/layer/attention_layer.h"

using namespace minfer;

TEST(Layer_TEST, attention_test)
{
    std::string ROOT_path =  std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    std::string input_path = ROOT_path + "atten_input.npy";
    std::string param0_path = ROOT_path + "atten_params_0.npy";
    std::string param1_path = ROOT_path + "atten_params_1.npy";
    std::string param2_path = ROOT_path + "atten_params_2.npy";
    std::string param3_path = ROOT_path + "atten_params_3.npy";
    std::string param_rms_path = ROOT_path + "atten_rms_params.npy";
    std::string output_path = ROOT_path + "atten_output.npy";

    Mat input = readMatFromNpy(input_path);
    Mat param0 = readMatFromNpy(param0_path); // q
    Mat param1 = readMatFromNpy(param1_path); // k
    Mat param2 = readMatFromNpy(param2_path); // v
    Mat param3 = readMatFromNpy(param3_path); // out
    Mat param_rms = readMatFromNpy(param_rms_path);
    Mat output = readMatFromNpy(output_path);

    const float rms_eps = 1e-6f;
    int d_model = 128;
    int num_heads = 8;
    int max_len = 256;
    int seq_len = 256;

//    FeedForwardLayerParams params
//     = {{0}, {1}, ActivateType::SILU, 128, 256, rms_eps, param_rms, param0, param2, param1};
    std::shared_ptr<AttentionLayerParams> layer_params(new AttentionLayerParams({0}, {1}, max_len, d_model, num_heads, num_heads, rms_eps, param_rms, param0, param1, param2, param3));
    auto layer = AttentionLayer::create(layer_params);

    std::vector<Mat*> inputs = {&input};

    Mat output_check;
    output_check.create(output.size.dims(), output.size.p, output.type());
    std::vector<Mat*> outputs = {&output_check};
    layer->forward(inputs, outputs, 0);

    std::cout<<"output.print(10) = "<<std::endl;
    output.print(10);

    std::cout<<"output_check.print(10) = "<<std::endl;
    output_check.print(10);
}