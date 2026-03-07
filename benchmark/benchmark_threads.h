#ifndef MINFER_BENCHMARK_THREADS_H
#define MINFER_BENCHMARK_THREADS_H

#ifdef _OPENMP
#include <omp.h>
#endif

namespace minfer {

inline int configure_benchmark_threads(int requested_threads = 0) {
#ifdef _OPENMP
    omp_set_dynamic(0);

    const int platform_threads = omp_get_num_procs();
    const int target_threads = requested_threads > 0
        ? requested_threads
        : (platform_threads > 0 ? platform_threads : 1);
    omp_set_num_threads(target_threads);
    return omp_get_max_threads();
#else
    (void)requested_threads;
    return 1;
#endif
}

inline int configure_benchmark_max_threads() {
    return configure_benchmark_threads();
}

} // namespace minfer

#endif // MINFER_BENCHMARK_THREADS_H
