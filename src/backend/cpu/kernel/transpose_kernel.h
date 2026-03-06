#ifndef MINFER_TRANSPOSE_KERNEL_H
#define MINFER_TRANSPOSE_KERNEL_H

#include <cstddef>

namespace minfer {
namespace cpu {

void transpose2d_kernel_blocked(const unsigned char* src,
                                unsigned char* dst,
                                int rows,
                                int cols,
                                size_t elem_size);

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_TRANSPOSE_KERNEL_H
