//
// Created by mzh on 2024/1/31.
//

#include "minfer/mat.h"
#include "minfer/basic_op.h"
#include "minfer/system.h"
#include "minfer/utils.h"

namespace minfer
{


static inline
MatShape make_strides(const MatShape& shape)
{
    MatShape strides(shape.size() - 2);
    size_t stride = 1;
    // 排除最后2个维度，他们作为内部循环，而其他的作为外部循环
    for (int i = (int)shape.size() - 2 - 1; i >= 0; i--)
    {
        strides[i] = stride;
        stride *= shape[i];
    }

    if (strides.empty())
        strides.push_back(1);
    return strides;
}

// naive impl, [M x K] x [K x N] = M x N
static inline
void gemm_impl_naive(const Mat& a, const Mat& b, Mat& c)
{
    MatShape shape_a = a.shape();
    MatShape shape_b = b.shape();

    // 目前不处理 K x KxN 这种情况。
    M_Assert(shape_a.size() >= 2 && shape_b.size() >= 2 && "Mat shapes on gemm function are miss matching!");

    // generate output shape with brodcast rule
    // need to handle When a and b shape is 1xK x KxN, or MxK x 1xN, or MxK x Kx1
    MatShape shape_c = get_gemm_shape(shape_a, shape_b);

    // M_Assert(shape_a.size() == shape_b.size() && "Two Mat dimension on gemm function are different!");
    int len_s = shape_a.size();
    M_Assert(len_s >= 2 && "Only multi-dimension Mat is supported for gemm function!");

    int M = shape_a.size() == 1 ? 1 : shape_a[shape_a.size() - 2];
    int K = shape_a[shape_a.size() - 1];
    int N = shape_b.size() == 1 ? 1 : shape_b[shape_b.size() - 1];

    M_Assert(K == shape_b[shape_b.size() - 2]); // 目前不支持有一个矩阵K为1的情况，后续考虑支持。
    M_Assert(a.type() == b.type());

    M_Assert(a.type() == DT_32F && "Currently only FP32 mat is supported!");

    // For dimension > 2, use numpy broadcasting rule for previous dimension.
    c = Mat(shape_c, DT_32F);

    size_t out_loop = len_s > 2 ? total(shape_c, 0, shape_c.size() - 2): 1;
    size_t step_a = M * K;
    size_t step_b = K * N;
    size_t step_c = M * N;

    MatShape stride_a = make_strides(shape_a);
    MatShape stride_b = make_strides(shape_b);
    MatShape stride_c = make_strides(shape_c);

    const float* pa = (const float*)a.data;
    const float* pb = (const float*)b.data;
    float* pc = (float*)c.data;

    for (int i = 0; i < out_loop; i++)
    {
        size_t tmp = i;
        std::vector<int> idx_c(stride_c.size());
        for (int d = 0; d < (int)stride_c.size(); d++)
        {
            idx_c[d] = tmp / stride_c[d];
            tmp %= stride_c[d];
        }

        // --- 广播到 a 的 batch index ---
        size_t lin_a = 0;
        for (int d = 0; d < (int)stride_a.size(); d++)
        {
            int coord = (shape_a[d] == 1) ? 0 : idx_c[d];
            lin_a += coord * stride_a[d];
        }

        // --- 广播到 b 的 batch index ---
        size_t lin_b = 0;
        for (int d = 0; d < (int)stride_b.size(); d++)
        {
            int coord = (shape_b[d] == 1) ? 0 : idx_c[d];
            lin_b += coord * stride_b[d];
        }

        const float* pai = lin_a * step_a + pa;
        const float* pbi = lin_b * step_b + pb;
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

    // 目前不处理 K x KxN 这种情况。
    M_Assert(shape_a.size() >= 2 && shape_b.size() >= 2 && "Mat shapes on gemm function are miss matching!");

    // generate output shape with brodcast rule
    // need to handle When a and b shape is 1xK x KxN, or MxK x 1xN, or MxK x Kx1
    MatShape shape_c;

    int index_a = shape_a.size() - 2;
    int index_b = shape_b.size() - 2;

    while (index_a > 0 || index_b > 0)
    {
        if (index_a > 0 && index_b > 0)
        {
            if (shape_a[index_a - 1] == shape_b[index_b - 1])
            {
                shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            }
            else if (shape_a[index_a - 1] == 1)
            {
                shape_c.insert(shape_c.begin(), shape_b[index_b - 1]);
            }
            else if (shape_b[index_b - 1] == 1)
            {
                shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            }
            else
            {
                M_Error(NULL, "Mat shapes on gemm function are miss matching!");
            }
            index_a--;
            index_b--;
        }
        else if (index_a > 0)
        {
            shape_c.insert(shape_c.begin(), shape_a[index_a - 1]);
            index_a--;
        }
        else if (index_b > 0)
        {
            shape_c.insert(shape_c.begin(), shape_b[index_b - 1]);
            index_b--;
        }
        else
            M_Error(NULL, "Mat shapes on gemm function are miss matching!");
    }

    // 如果有一个维度为1维度，说明 出现 MxK x K = M的情况
    if (shape_a.size() == 1 || shape_b.size() == 1)
    {
        if (shape_a.size() == 1 && shape_b.size() == 1)
        {
            shape_c = {1}; // 处理 1xK x Kx1 = 1的情况
        }
    }
    else
    {
        shape_c.push_back(shape_a[shape_a.size() - 2]);
        shape_c.push_back(shape_b[shape_b.size() - 2]);
    }

    // M_Assert(shape_a.size() == shape_b.size() && "Two Mat dimension on gemm function are different!");
    int len_s = shape_a.size();
    M_Assert(len_s >= 2 && "Only multi-dimension Mat is supported for gemm function!");

    int M = shape_a.size() == 1 ? 1 : shape_a[shape_a.size() - 2];
    int K = shape_a[shape_a.size() - 1];
    int N = shape_b.size() == 1 ? 1 : shape_b[shape_b.size() - 2];

    if (K != shape_b[shape_b.size() - 1])
    M_Assert(K == shape_b[shape_b.size() - 1]); // 目前不支持有一个矩阵K为1的情况，后续考虑支持。
    M_Assert(a.type() == b.type());

    M_Assert(a.type() == DT_32F && "Currently only FP32 mat is supported!");

    // For dimension > 2, use numpy broadcasting rule for previous dimension.
    c = Mat(shape_c, DT_32F);

    size_t out_loop = len_s > 2 ? total(shape_c, 0, shape_c.size() - 2): 1;
    size_t step_a = M * K;
    size_t step_b = K * N;
    size_t step_c = M * N;

    MatShape stride_a = make_strides(shape_a);
    MatShape stride_b = make_strides(shape_b);
    MatShape stride_c = make_strides(shape_c);
    const float* pa = (const float*)a.data;
    const float* pb = (const float*)b.data;
    float* pc = (float*)c.data;

    for (int i = 0; i < out_loop; i++)
    {
        size_t tmp = i;
        std::vector<int> idx_c(stride_c.size());
        for (int d = 0; d < (int)stride_c.size(); d++)
        {
            idx_c[d] = tmp / stride_c[d];
            tmp %= stride_c[d];
        }

        // --- 广播到 a 的 batch index ---
        size_t lin_a = 0;
        for (int d = 0; d < (int)stride_a.size(); d++)
        {
            int coord = (shape_a[d] == 1) ? 0 : idx_c[d];
            lin_a += coord * stride_a[d];
        }

        // --- 广播到 b 的 batch index ---
        size_t lin_b = 0;
        for (int d = 0; d < (int)stride_b.size(); d++)
        {
            int coord = (shape_b[d] == 1) ? 0 : idx_c[d];
            lin_b += coord * stride_b[d];
        }

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
        Mat aT;
        Mat bT;

        if (transA)
            aT = transpose(a);
        else
            aT = a;

        if (transB)
            bT = transpose(b);
        else
            bT = b;

        gemm_impl_naive(aT, bT, out);
    }

    return out;
}

}