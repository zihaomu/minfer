//
// Created by mzh on 2024/10/29.
//

#include "minfer.h"
#include "gtest/gtest.h"
#include "../src/backend/cpu/layer/feed_forward.h"

using namespace minfer;

TEST(Layer_TEST, feed_forward_test)
{
    std::string ROOT_path = std::string(M_ROOT_PATH) + "/test/layers/test_data/data/";
    std::string input_path = ROOT_path + "ffn_input.npy";
    std::string param0_path = ROOT_path + "ffn_params_0.npy";
    std::string param1_path = ROOT_path + "ffn_params_1.npy";
    std::string param2_path = ROOT_path + "ffn_params_2.npy";
    std::string param_rms_path = ROOT_path + "ffn_rms_params.npy";
    std::string output_path = ROOT_path + "ffn_output.npy";

    Mat input = readMatFromNpy(input_path);
    Mat param0 = readMatFromNpy(param0_path); // gate
    Mat param1 = readMatFromNpy(param1_path); // down
    Mat param2 = readMatFromNpy(param2_path); // up
    Mat param_rms = readMatFromNpy(param_rms_path);
    Mat output = readMatFromNpy(output_path);

    const float rms_eps = 1e-6f;
    FeedForwardLayerParams params
     = {{0}, {1}, ActivateType::SILU, 128, 256, rms_eps, param_rms, param0, param2, param1};
    std::shared_ptr<FeedForwardLayerParams> layer_params(new FeedForwardLayerParams({0}, {1}, ActivateType::SILU, 128, 256, rms_eps, param_rms, param0, param2, param1));
    auto layer = FeedForwardLayer::create(layer_params);

    std::vector<Mat*> inputs = {&input};

    Mat output_check;
    output_check.create(output.size.dims(), output.size.p, output.type());
    std::vector<Mat*> outputs = {&output_check};
    layer->forward(inputs, outputs, 0);

    std::cout<<"output.print(10) = "<<std::endl;
    output.print(10);

    std::cout<<"output_check.print(10) = "<<std::endl;
    output_check.print(10);

    double v = norm(output, output_check, NORM_L1);
    std::cout<<"v = "<<v<<std::endl;
    M_Assert(v < 12);
}