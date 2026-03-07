#include "gtest/gtest.h"

#include "../benchmark/benchmark_threads.h"
#include "backend/cpu/kernel/openmp_utils.h"

TEST(BenchmarkThreads, configures_platform_max_threads_by_default) {
#ifdef _OPENMP
    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();
#endif

    const int active_threads = minfer::configure_benchmark_threads();

#ifdef _OPENMP
    const int platform_threads = omp_get_num_procs();
    EXPECT_GE(platform_threads, 1);
    EXPECT_EQ(active_threads, platform_threads);
    EXPECT_EQ(omp_get_max_threads(), platform_threads);

    omp_set_dynamic(prev_dynamic);
    omp_set_num_threads(prev_threads);
#else
    EXPECT_EQ(active_threads, 1);
#endif
}

TEST(BenchmarkThreads, honors_explicit_thread_count) {
#ifdef _OPENMP
    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();
    const int platform_threads = omp_get_num_procs();
    const int requested_threads = platform_threads > 1 ? platform_threads - 1 : 1;
#endif

    const int active_threads =
#ifdef _OPENMP
        minfer::configure_benchmark_threads(requested_threads);
#else
        minfer::configure_benchmark_threads(4);
#endif

#ifdef _OPENMP
    EXPECT_EQ(active_threads, requested_threads);
    EXPECT_EQ(omp_get_max_threads(), requested_threads);

    omp_set_dynamic(prev_dynamic);
    omp_set_num_threads(prev_threads);
#else
    EXPECT_EQ(active_threads, 1);
#endif
}

TEST(BenchmarkThreads, skips_parallel_region_for_tiny_outer_loops) {
#ifdef _OPENMP
    const int prev_dynamic = omp_get_dynamic();
    const int prev_threads = omp_get_max_threads();

    omp_set_dynamic(0);
    omp_set_num_threads(8);

    EXPECT_FALSE(minfer::cpu::should_parallelize_1d_loop(1, 512u * 2048u, 1LL << 16, 1));
    EXPECT_FALSE(minfer::cpu::should_parallelize_1d_loop(8, 512, 1LL << 14, 2));
    EXPECT_TRUE(minfer::cpu::should_parallelize_1d_loop(128, 512, 1LL << 14, 2));

    omp_set_dynamic(prev_dynamic);
    omp_set_num_threads(prev_threads);
#else
    EXPECT_FALSE(minfer::cpu::should_parallelize_1d_loop(128, 512, 1LL << 14, 2));
#endif
}
