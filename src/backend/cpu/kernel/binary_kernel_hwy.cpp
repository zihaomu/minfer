#include "binary_kernel_hwy.h"
#include "openmp_utils.h"

#include "hwy/highway.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {
namespace cpu {

namespace hn = hwy::HWY_NAMESPACE;

namespace {

inline float apply_scalar(BinaryKernelOp op, float lhs, float rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
    }

    return 0.0f;
}

}  // namespace

void binary_broadcast_hwy(BinaryKernelOp op,
                          const float* lhs,
                          size_t lhs_outer_stride,
                          size_t lhs_inner_stride,
                          const float* rhs,
                          size_t rhs_outer_stride,
                          size_t rhs_inner_stride,
                          float* out,
                          size_t outer,
                          size_t inner)
{
    const hn::ScalableTag<float> d;
    const size_t lanes = static_cast<size_t>(hn::Lanes(d));

    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* lhs_row = lhs + outer_i * lhs_outer_stride;
        const float* rhs_row = rhs + outer_i * rhs_outer_stride;
        float* out_row = out + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride == 1 && rhs_inner_stride == 1)
        {
            for (; inner_idx + lanes <= inner; inner_idx += lanes)
            {
                const auto lhs_vec = hn::LoadU(d, lhs_row + inner_idx);
                const auto rhs_vec = hn::LoadU(d, rhs_row + inner_idx);
                auto out_vec = lhs_vec;

                switch (op)
                {
                    case BinaryKernelOp::Add:
                        out_vec = hn::Add(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Sub:
                        out_vec = hn::Sub(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Mul:
                        out_vec = hn::Mul(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Div:
                        out_vec = hn::Div(lhs_vec, rhs_vec);
                        break;
                }

                hn::StoreU(out_vec, d, out_row + inner_idx);
            }
        }
        else if (lhs_inner_stride == 0 && rhs_inner_stride == 1)
        {
            for (; inner_idx + lanes <= inner; inner_idx += lanes)
            {
                const auto lhs_vec = hn::Set(d, lhs_row[0]);
                const auto rhs_vec = hn::LoadU(d, rhs_row + inner_idx);
                auto out_vec = lhs_vec;

                switch (op)
                {
                    case BinaryKernelOp::Add:
                        out_vec = hn::Add(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Sub:
                        out_vec = hn::Sub(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Mul:
                        out_vec = hn::Mul(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Div:
                        out_vec = hn::Div(lhs_vec, rhs_vec);
                        break;
                }

                hn::StoreU(out_vec, d, out_row + inner_idx);
            }
        }
        else if (lhs_inner_stride == 1 && rhs_inner_stride == 0)
        {
            for (; inner_idx + lanes <= inner; inner_idx += lanes)
            {
                const auto lhs_vec = hn::LoadU(d, lhs_row + inner_idx);
                const auto rhs_vec = hn::Set(d, rhs_row[0]);
                auto out_vec = lhs_vec;

                switch (op)
                {
                    case BinaryKernelOp::Add:
                        out_vec = hn::Add(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Sub:
                        out_vec = hn::Sub(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Mul:
                        out_vec = hn::Mul(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Div:
                        out_vec = hn::Div(lhs_vec, rhs_vec);
                        break;
                }

                hn::StoreU(out_vec, d, out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const float rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar(op, lhs_val, rhs_val);
        }
    }
}

}  // namespace cpu
}  // namespace minfer
