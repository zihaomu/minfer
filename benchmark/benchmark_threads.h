#ifndef MINFER_BENCHMARK_THREADS_H
#define MINFER_BENCHMARK_THREADS_H

#include "minfer/parallel.h"

namespace minfer {

inline int configure_benchmark_threads(int requested_threads = 0) {
    const int platform_threads = parallel_get_num_procs();
    const int target_threads = requested_threads > 0
        ? requested_threads
        : (platform_threads > 0 ? platform_threads : 1);
    parallel_set_num_threads(target_threads);
    return parallel_get_num_threads();
}

inline int configure_benchmark_max_threads() {
    return configure_benchmark_threads();
}

} // namespace minfer

#endif // MINFER_BENCHMARK_THREADS_H
