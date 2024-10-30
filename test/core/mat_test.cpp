//
// Created by mzh on 2024/10/29.
//

#include "minfer.h"
#include "gtest/gtest.h"

using namespace minfer;

// TODO add test element equal check. compare two mat, or compare mat and scalar.
TEST(Mat_TEST, loadNpy)
{
    // try to get path from system
    std::string path = "/Users/mzh/work/my_project/minfer/test/layers/test_data/data/random10x12.npy";

    Mat m = readMatFromNpy(path);

    m.print();

    float * data = (float *)m.data;

    float a0 = 0.5488135;
    float a1 = 0.71518934;
    float a2 = 0.56804454;

    M_Assert(data[0] == a0);
    M_Assert(data[1] == a1);
    M_Assert(data[12] == a2);
}
