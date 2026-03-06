#ifndef MINFER_ROPE_KERNEL_H
#define MINFER_ROPE_KERNEL_H

namespace minfer {
namespace cpu {

void rope_kernel_inplace(float* q,
                         float* k,
                         int seq_len,
                         int start_pos,
                         int head_count,
                         int head_count_kv,
                         int head_dim,
                         float freq_base = 10000.0f);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_ROPE_KERNEL_H
