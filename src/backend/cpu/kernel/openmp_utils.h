#ifndef MINFER_BACKEND_CPU_KERNEL_OPENMP_UTILS_H
#define MINFER_BACKEND_CPU_KERNEL_OPENMP_UTILS_H

#include <cstddef>
#include "minfer/parallel.h"

namespace minfer {
namespace cpu {

inline bool should_parallelize_1d_loop(size_t trip_count,
                                       size_t work_per_item,
                                       long long min_total_work,
                                       int min_items_per_thread = 1)
{
    if (parallel_in_parallel())
    {
        return false;
    }

    if (trip_count <= 1)
    {
        return false;
    }

    const int max_threads = parallel_get_num_threads();
    if (max_threads <= 1)
    {
        return false;
    }

    const size_t min_trip_count = static_cast<size_t>(max_threads) *
                                  static_cast<size_t>(min_items_per_thread > 1 ? min_items_per_thread : 1);
    if (trip_count < min_trip_count)
    {
        return false;
    }

    const auto total_work = static_cast<unsigned long long>(trip_count) *
                            static_cast<unsigned long long>(work_per_item);
    const auto required_work = static_cast<unsigned long long>(min_total_work > 0 ? min_total_work : 0);
    return total_work >= required_work;
}

}  // namespace cpu
}  // namespace minfer

#endif  // MINFER_BACKEND_CPU_KERNEL_OPENMP_UTILS_H
