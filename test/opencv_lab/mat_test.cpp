//
// Created by mzh on 2024/2/1.
//

#include "mat_test/mat.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;
using namespace opencv_lab;

TEST(OpenCVMat, create_1)
{
    Mat m0 = Mat();

    Mat m1 = Mat({1, 3, 4}, 0);

    assert(m1.u->refcount == 1);
//    m0 = m1;

//    assert(m0.u->refcount == 2);
    std::cout<<"Tensor 1"<<std::endl;
//    CV_SINGLETON_LAZY_INIT(A, new A());
//    CV_SINGLETON_LAZY_INIT(A, new A());
//    return true;
}