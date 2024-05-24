//
// Created by mzh on 2024/4/1.
//

#include "minfer.h"
#include "gtest/gtest.h"

using namespace minfer;

TEST(Net_TEST, simple_net_test)
{
//    class LayerParams
//    {
//    public:
//        int layerId = -1;
//        std::string name;
//        LayerType type;
//        std::vector<int> inputIndex;
//        std::vector<int> outputIndex;
//    };

//    Mat a;
////    a = Mat({2, 3, 4}, 20.f);
//    Mat* b = &a;
//    a = Mat({2, 3, 5, 4}, 20.f);
//    Mat c;
//    b = &c;
//
//    Mat d = Mat({2, 3, 4}, 20.f);
//    c = d;

    float a = 20.f;
    int intValue = *reinterpret_cast<int*>(&a);

    std::cout << "Float value: " << a << std::endl;
    std::cout << "Reinterpreted int value: " << intValue << std::endl;


    std::shared_ptr<LayerParams> input0 = std::shared_ptr<LayerParams>(new LayerParams(0, "inp_0", LayerType::Input, {0}, {1}));
    std::shared_ptr<LayerParams> input1 = std::shared_ptr<LayerParams>(new LayerParams(1, "inp_1", LayerType::Input, {2}, {3}));
    std::shared_ptr<LayerParams> add = std::shared_ptr<LayerParams>(new LayerParams(2, "add", LayerType::Add, {1,3}, {4}));
    std::shared_ptr<LayerParams> out = std::shared_ptr<LayerParams>(new LayerParams(3, "out", LayerType::Output, {4}, {5}));

    Net net;
    net.createLayer(input0);
    net.createLayer(input1);
    net.createLayer(add);
    net.createLayer(out);

    float f20 = 20.f;
    float f30 = 30.f;
    float f50 = 50.f;
    Mat inpM1 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f20));
    Mat inpM2 = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f30));
    Mat outM  = Mat({2, 3, 4}, DT_32F, reinterpret_cast<int&>(f50));

    inpM1.print();
    inpM2.print();

    net.setInput(inpM1, 0);
    net.setInput(inpM2, 2);

    net.init();
    Mat outMat = net.forward();

    outM.print();
}


TEST(Net_TEST, net_tiny_llama)
{
    std::cout<<"print test on net_tiny_llama"<<std::endl;
    Net net;
    net.readNet("tinyllamas-stories-260k-f32.gguf");

    // TODO add the forward type.
}