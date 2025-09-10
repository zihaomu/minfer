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
    Mat up = readMatFromNpy(param0_path); // gate
    Mat gate = readMatFromNpy(param1_path); // down
    Mat down = readMatFromNpy(param2_path); // up
    Mat param_rms = readMatFromNpy(param_rms_path);
    Mat output = readMatFromNpy(output_path);

    const float rms_eps = 1e-6f;
    std::shared_ptr<FeedForwardLayerParams> layer_params(new FeedForwardLayerParams({0}, {1}, ActivateType::SILU, 128, 256, rms_eps, param_rms, gate, up, down));
    auto layer = FeedForwardLayer::create(layer_params);

    std::vector<Mat*> inputs = {&input};

    Mat output_check;
    output_check.create(output.size.dims(), output.size.p, output.type());
    std::vector<Mat*> outputs = {&output_check};
    layer->forward(inputs, outputs);

    std::cout<<"output.print(10) = "<<std::endl;
    output.print(10);

    std::cout<<"output_check.print(10) = "<<std::endl;
    output_check.print(10);

    double mean_l1 = norm(output, output_check, NORM_L1) / output.total();
    double rel_l2_a  = norm(output, output_check, NORM_L2);
    double rel_l2_b = norm(output_check, NORM_L2) + 1e-12;
    double rel_l2 = rel_l2_a / rel_l2_b;
    double max_err = norm(output, output_check, NORM_INF);

    std::cout << "mean L1 = " << mean_l1
              << ", relative L2 = " << rel_l2
              << ", max abs = " << max_err << std::endl;

    M_Assert(mean_l1 < 0.45);
    M_Assert(rel_l2  < 1e-9);
    M_Assert(max_err < 2);

    // double v = norm(output, output_check, NORM_L1);
    // std::cout<<"v = "<<v<<std::endl;
    // M_Assert(v < 12);
}