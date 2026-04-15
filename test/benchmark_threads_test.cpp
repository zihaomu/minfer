#include "gtest/gtest.h"

#include "../benchmark/benchmark_threads.h"
#include "backend/cpu/kernel/openmp_utils.h"
#include "minfer/parallel.h"
#include "minfer/system.h"

TEST(BenchmarkThreads, configures_platform_max_threads_by_default) {
    const int prev_threads = minfer::get_num_threads();

    const int active_threads = minfer::configure_benchmark_threads();
    const int platform_threads = minfer::parallel_get_num_procs();
    const int expected_threads = platform_threads > 0 ? platform_threads : 1;

    EXPECT_GE(platform_threads, 1);
    EXPECT_EQ(active_threads, expected_threads);
    EXPECT_EQ(minfer::get_num_threads(), expected_threads);

    minfer::set_num_threads(prev_threads);
}

TEST(BenchmarkThreads, honors_explicit_thread_count) {
    const int prev_threads = minfer::get_num_threads();
    const int platform_threads = minfer::parallel_get_num_procs();
    const int requested_threads = platform_threads > 1 ? platform_threads - 1 : 1;
    const int active_threads = minfer::configure_benchmark_threads(requested_threads);

    EXPECT_EQ(active_threads, requested_threads);
    EXPECT_EQ(minfer::get_num_threads(), requested_threads);

    minfer::set_num_threads(prev_threads);
}

TEST(BenchmarkThreads, skips_parallel_region_for_tiny_outer_loops) {
    const int prev_threads = minfer::get_num_threads();
    minfer::set_num_threads(8);
    EXPECT_FALSE(minfer::cpu::should_parallelize_1d_loop(1, 512u * 2048u, 1LL << 16, 1));
    EXPECT_FALSE(minfer::cpu::should_parallelize_1d_loop(8, 512, 1LL << 14, 2));
    EXPECT_TRUE(minfer::cpu::should_parallelize_1d_loop(128, 512, 1LL << 14, 2));

    minfer::set_num_threads(prev_threads);
}
