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

TEST(Net_TEST, multi_head_transform_test)
{
    // 参数目前为空
//    int head_count;
//    int head_count_kv;
//    Mat _q;
//    Mat _k;
//    Mat _v;
//    Mat _out;
//
//    // 需要增加参数，减少layer和layer name的要求。
//    std::vector<std::shared_ptr<LayerParams> > layers = {
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::Input, {0}, {1})),
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::RMSNorm, {1}, {2})),
//            std::shared_ptr<LayerParams>(new AttentionLayerParams(LayerType::Attention,{2}, {3}, head_count,
//                                                                  head_count_kv, _q, _k, _v, _out)),
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::Add, {1,3}, {4})),
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::RMSNorm, {4}, {5})),
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::FFN, {5}, {6})),
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::Add, {4,6}, {7})),
//            std::shared_ptr<LayerParams>(new LayerParams(LayerType::Output, {7}, {8})),
//    };
//
//    Net net;
//    net.createNet(layers);
//
//    float f20 = 20.f;
//    float f30 = 30.f;
//    float f50 = 50.f;
//    Mat inpM1 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f20));
//    Mat inpM2 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f30));
//    Mat outM  = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f50));
//
//    inpM1.print();
//    inpM2.print();
//
//    net.setInput(inpM1, 0);
//    net.setInput(inpM2, 2);
//
//    net.init();
//    Mat outMat = net.forward();
//
//    outM.print();
}

TEST(Net_TEST, net_tiny_llama)
{
    std::cout<<"print test on net_tiny_llama"<<std::endl;
    Net net;
    net.readNet("/Users/mzh/work/models/llm_model/lite_oute_llama/Lite-Oute-1-65M-FP16.gguf");

    // TODO add the forward type.
}