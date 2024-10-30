//
// Created by mzh on 2024/1/31.
//

#include "minfer/mat.h"
#include "minfer/basic_op.h"
#include "minfer/system.h"
#include "minfer/utils.h"

namespace minfer
{

// naive impl, [M x K] x [K x N] = M x N
static inline
void gemm_impl_naive(const Mat& a, const Mat& b, Mat& c)
{
    MatShape shape_a = a.shape();
    MatShape shape_b = b.shape();

    M_Assert(shape_a.size() == shape_b.size() && "Two Mat dimension on gemm function are different!");
    int len_s = shape_a.size();
    M_Assert(len_s >= 2 && "Only multi-dimension Mat is supported for gemm function!");

    int M = shape_a[len_s - 2];
    int K = shape_a[len_s - 1];
    int N = shape_b[len_s - 1];

    M_Assert(K == shape_b[len_s - 2]);
    M_Assert(a.type() == b.type());

    M_Assert(a.type() == DT_32F && "Currently only FP32 mat is supported!");

    int i = len_s - 3;
    while (i > 0)
    {
        M_Assert(shape_a[i] == shape_b[i] && "Mat shapes on gemm function are miss matching!");
        i--;
    }

    // For dimension > 2, use numpy broadcasting rule for previous dimension.
    // Add broadcasting rule here.
    MatShape outShape(shape_a); // M x N
    outShape[len_s - 1] = N;

    c = Mat(outShape, DT_32F);

    size_t out_loop = len_s > 2 ? total(shape_a, 0, len_s - 3): 1;
    size_t step_a = M * K;
    size_t step_b = K * N;
    size_t step_c = M * N;

    const float* pa = (const float*)a.data;
    const float* pb = (const float*)b.data;
    float* pc = (float*)c.data;

    for (int i = 0; i < out_loop; i++)
    {
        const float* pai = i * step_a + pa;
        const float* pbi = i * step_b + pb;
        float* pci = i * step_c + pc;

        // TODO optimize the gemm kernel, the following is naive implementation.
        for (int m = 0; m < M; m++)
        {
            const float* paim = pai + K * m;
            float* pcim = pci + N * m;

            for (int n = 0; n < N; n++)
            {
                const float* pbin = pbi + n;
                float sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += paim[k] * pbin[k * N];
                }

                pcim[n] = sum;
            }
        }
    }
}

// Mat b is not transposed! [M x K] x [N x K] = M x N
static inline
void gemm_impl_row(const Mat& a, const Mat& b, Mat& c)
{
    MatShape shape_a = a.shape();
    MatShape shape_b = b.shape();

    // TODO, support the different shape gemm!
    M_Assert(shape_a.size() == shape_b.size() && "Two Mat dimension on gemm function are different!");
    int len_s = shape_a.size();
    M_Assert(len_s >= 2 && "Only multi-dimension Mat is supported for gemm function!");

    int M = shape_a[len_s - 2];
    int K = shape_a[len_s - 1];
    int N = shape_b[len_s - 2];

    M_Assert(K == shape_b[len_s - 1]);
    M_Assert(a.type() == b.type());
    M_Assert(a.type() == DT_32F && "Currently only FP32 mat is supported!");

    int i = len_s - 3;
    while (i > 0)
    {
        M_Assert(shape_a[i] == shape_b[i] && "Mat shapes on gemm function are miss matching!");
        i--;
    }

    MatShape outShape(shape_a); // M x N
    outShape[len_s - 1] = N;

    c = Mat(outShape, DT_32F);

    size_t out_loop = len_s > 2 ? total(shape_a, 0, len_s - 3): 1;
    size_t step_a = M * K;
    size_t step_b = K * N;
    size_t step_c = M * N;

    const float* pa = (const float*)a.data;
    const float* pb = (const float*)b.data;
    float* pc = (float*)c.data;

    for (int i = 0; i < out_loop; i++)
    {
        const float* pai = i * step_a + pa;
        const float* pbi = i * step_b + pb;
        float* pci = i * step_c + pc;

        // TODO optimize the gemm kernel, the following is naive implementation.
        for (int m = 0; m < M; m++)
        {
            const float* paim = pai + K * m;
            float* pcim = pci + N * m;

            for (int n = 0; n < N; n++)
            {
                const float* pbin = pbi + n * K;
                float sum = 0;
                for (int k = 0; k < K; k++)
                {
                    sum += paim[k] * pbin[k];
                }

                pcim[n] = sum;
            }
        }
    }
}

// implementation of rox x column
// TODO gemm support the different shape.
Mat gemm(const Mat& a, const Mat& b, bool transA, bool transB)
{
    Mat out;
    if (transA == false && transB == true)
    {
        gemm_impl_row(a, b, out);
    }
    else
    {
        Mat aT = a;
        Mat bT = b;

        if (transA)
            transpose(a, aT);

        if (transB)
            transpose(b, bT);

        gemm_impl_naive(aT, bT, out);
    }

    return out;
}

}