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
    std::string path = std::string(M_ROOT_PATH) + "/test/core/test_data/data/random10x12.npy";

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

TEST(Mat_TEST, Op_Test)
{
    Mat m3 = Mat({2, 3, 4}, DT_32F);
    m3.setTo(3.0f);

    Mat m2 = Mat({2, 3, 4}, DT_32F);
    m2.setTo(2.0f);

    Mat m4 = Mat({2, 3, 4}, DT_8S);
    m4.setTo(3.0f);

    Mat m5 = Mat({2, 3, 4}, DT_8S);
    m5.setTo(2.0f);

    m3.print();
    m2.print();

    m4.print();
    m5.print();

//    Mat m = m3 - m2;
//    m.print();

    Mat m22 = m4 - m5;
    m22.print();

//    double v = norm(m, m2, NORM_L1);
//    M_Assert(v < 1e-4);
}


TEST(Mat_TEST, Op_Test2)
{
    std::vector<int> test_shape = {1, 4, 5};
    Mat m3 = Mat(test_shape, DT_32F);
    m3.setTo(3.0f);

    Mat m4 = m3 - 1.f;

    Mat m2 = Mat(test_shape, DT_32F);
    m2.setTo(2.0f);

    double v = norm(m4, m2, NORM_L1);
    M_Assert(v < 1e-4);

    Mat m5 = (1 - m3) * 3;
    Mat m6 = Mat(test_shape, DT_32F);
    m6.setTo(-2.0f * 3);

    double v2 = norm(m5, m6, NORM_L1);
    M_Assert(v2 < 1e-4);

}

TEST(Mat_TEST, transposeND)
{
    Mat m = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_0_i.npy");
    Mat m_cheker_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_1_o.npy");
    Mat m_cheker_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_2_o.npy");
    Mat m_cheker_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_3_o.npy");
    Mat m_cheker_4 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/trans_3d_4_o.npy");

    Mat out0 = transposeND(m, {2, 1, 0});
    Mat out1 = transposeND(m, {0, 2, 1});
    Mat out2 = transposeND(m, {2, 0, 1});
    Mat out3 = transposeND(m, {1, 0, 2});

//    std::cout<<"out.print(10) = "<<std::endl;
//    m.print(30);
//    out0.print(30);
//    m_cheker_1.print(30);
//    out1.print(30);
//    m_cheker_2.print(30);
//    out2.print(30);
//    m_cheker_3.print(30);
//    out3.print(30);
//    m_cheker_4.print(30);

    double v0 = norm(out0, m_cheker_1, NORM_L1);
    double v1 = norm(out1, m_cheker_2, NORM_L1);
    double v2 = norm(out2, m_cheker_3, NORM_L1);
    double v3 = norm(out3, m_cheker_4, NORM_L1);

    M_Assert(v0 < 1e-4);
    M_Assert(v1 < 1e-4);
    M_Assert(v2 < 1e-4);
    M_Assert(v3 < 1e-4);
}


TEST(Mat_TEST, mat_mul)
{
    Mat mi_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_0_i.npy");
    Mat mi_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_i.npy");
    Mat mi_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_i.npy");
    Mat mi_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_i.npy");
    Mat mi_4 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_i.npy");
    Mat mo_checker_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_o.npy");
    Mat mo_checker_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_o.npy");
    Mat mo_checker_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_o.npy");
    Mat mo_checker_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_o.npy");


    Mat out0 = gemm(mi_0, mi_1);
    Mat out1 = gemm(mi_0, mi_2, false, true);
    Mat out2 = gemm(mi_0, mi_3, true, false);
    Mat out3 = gemm(mi_0, mi_4, true, true);

    double v0 = norm(out0, mo_checker_0, NORM_L1);
    double v1 = norm(out1, mo_checker_1, NORM_L1);
    double v2 = norm(out2, mo_checker_2, NORM_L1);
    double v3 = norm(out3, mo_checker_3, NORM_L1);

    std::cout<<"norm 0= "<<v0<<std::endl;
    std::cout<<"norm 1= "<<v1<<std::endl;
    std::cout<<"norm 2= "<<v2<<std::endl;
    std::cout<<"norm 3= "<<v3<<std::endl;

    M_Assert(v0 < 1e-3); // m_cheker_1 is empty, TODO, check why this happen.
    M_Assert(v1 < 1e-3);
    M_Assert(v2 < 1e-3);
    M_Assert(v3 < 1e-3);
}

TEST(Mat_TEST, mat_mul_broad_cast)
{
    Mat mi_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_0_i.npy");
    Mat mi_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_i.npy");
    Mat mi_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_i.npy");
    Mat mi_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_i.npy");
    Mat mi_4 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_i.npy");
    Mat mo_checker_0 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_1_o.npy");
    Mat mo_checker_1 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_2_o.npy");
    Mat mo_checker_2 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_3_o.npy");
    Mat mo_checker_3 = readMatFromNpy(std::string(M_ROOT_PATH) + "/test/core/test_data/data/matmul_4_o.npy");


    Mat out0 = gemm(mi_0, mi_1);
    Mat out1 = gemm(mi_0, mi_2, false, true);
    Mat out2 = gemm(mi_0, mi_3, true, false);
    Mat out3 = gemm(mi_0, mi_4, true, true);

    double v0 = norm(out0, mo_checker_0, NORM_L1);
    double v1 = norm(out1, mo_checker_1, NORM_L1);
    double v2 = norm(out2, mo_checker_2, NORM_L1);
    double v3 = norm(out3, mo_checker_3, NORM_L1);

    std::cout<<"norm 0= "<<v0<<std::endl;
    std::cout<<"norm 1= "<<v1<<std::endl;
    std::cout<<"norm 2= "<<v2<<std::endl;
    std::cout<<"norm 3= "<<v3<<std::endl;

    M_Assert(v0 < 1e-3); // m_cheker_1 is empty, TODO, check why this happen.
    M_Assert(v1 < 1e-3);
    M_Assert(v2 < 1e-3);
    M_Assert(v3 < 1e-3);
}

TEST(Mat_TEST, data_convert_fp16_to_fp32)
{
    std::vector<int> test_shape = {1, 4, 5};
    Mat m0 = Mat(test_shape, DT_32F);
    m0.setTo(3.0f);

    Mat m16;
    m0.convertTo(m16, DT_16F);

    Mat m32;
    m16.convertTo(m32, DT_32F);

    double v0 = norm(m32, m0, NORM_L1);
    M_Assert(v0 < 1e-3);
}

