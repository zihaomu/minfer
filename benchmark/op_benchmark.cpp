#include "minfer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace minfer;

namespace {

struct BenchmarkOptions
{
    int warmup = 10;
    int iters = 100;
    int batch = 4;
    int seq_len = 128;
    int hidden = 1024;
    int gemm_out = 0;
    int head_count = 8;
    int head_count_kv = 4;
    int threads = 0;
    bool gemm_only = false;
};

BenchmarkOptions parse_args(int argc, char** argv)
{
    BenchmarkOptions opts;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto need_value = [&](const std::string& name) {
            if (i + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + name);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--warmup")
            opts.warmup = std::stoi(need_value(arg));
        else if (arg == "--iters")
            opts.iters = std::stoi(need_value(arg));
        else if (arg == "--batch")
            opts.batch = std::stoi(need_value(arg));
        else if (arg == "--seq-len")
            opts.seq_len = std::stoi(need_value(arg));
        else if (arg == "--hidden")
            opts.hidden = std::stoi(need_value(arg));
        else if (arg == "--gemm-out")
            opts.gemm_out = std::stoi(need_value(arg));
        else if (arg == "--head-count")
            opts.head_count = std::stoi(need_value(arg));
        else if (arg == "--head-count-kv")
            opts.head_count_kv = std::stoi(need_value(arg));
        else if (arg == "--threads")
            opts.threads = std::stoi(need_value(arg));
        else if (arg == "--gemm-only")
            opts.gemm_only = true;
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --warmup <n>\n"
                << "  --iters <n>\n"
                << "  --batch <n>\n"
                << "  --seq-len <n>\n"
                << "  --hidden <n>\n"
                << "  --gemm-out <n> (default: hidden * 4)\n"
                << "  --head-count <n>\n"
                << "  --head-count-kv <n>\n"
                << "  --threads <n>\n"
                << "  --gemm-only\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (opts.warmup < 0 || opts.iters <= 0 || opts.batch <= 0 || opts.seq_len <= 0 || opts.hidden <= 0)
    {
        throw std::runtime_error("All benchmark sizes must be positive");
    }
    if (opts.head_count <= 0 || opts.head_count_kv <= 0)
    {
        throw std::runtime_error("Head counts must be positive");
    }
    if (opts.gemm_out < 0)
    {
        throw std::runtime_error("--gemm-out must be non-negative");
    }
    if (opts.hidden % opts.head_count != 0)
    {
        throw std::runtime_error("--hidden must be divisible by --head-count");
    }
    if (opts.gemm_out == 0)
    {
        opts.gemm_out = opts.hidden * 4;
    }

    return opts;
}

int configured_threads(const BenchmarkOptions& opts)
{
#ifdef _OPENMP
    if (opts.threads > 0)
    {
        omp_set_dynamic(0);
        omp_set_num_threads(opts.threads);
    }
    return omp_get_max_threads();
#else
    (void)opts;
    return 1;
#endif
}

void fill_random(Mat& mat, std::mt19937& rng)
{
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float* data = reinterpret_cast<float*>(mat.data);
    for (size_t i = 0; i < mat.total(); ++i)
    {
        data[i] = dist(rng);
    }
}

double benchmark_ms(const BenchmarkOptions& opts, const std::function<void()>& fn)
{
    for (int i = 0; i < opts.warmup; ++i)
    {
        fn();
    }

    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < opts.iters; ++i)
    {
        fn();
    }
    const auto end = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    return total_ms / opts.iters;
}

struct BenchmarkStats
{
    double avg_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
};

double percentile_ms(std::vector<double> values, double percentile)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::max(0.0, std::min(1.0, percentile));
    const size_t rank = static_cast<size_t>(std::ceil(clamped * static_cast<double>(values.size())));
    const size_t index = rank == 0 ? 0 : std::min(values.size() - 1, rank - 1);
    return values[index];
}

BenchmarkStats benchmark_stats(const BenchmarkOptions& opts, const std::function<void()>& fn)
{
    for (int i = 0; i < opts.warmup; ++i)
    {
        fn();
    }

    std::vector<double> samples_ms;
    samples_ms.reserve(static_cast<size_t>(opts.iters));

    for (int i = 0; i < opts.iters; ++i)
    {
        const auto begin = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }

    double total_ms = 0.0;
    for (double sample_ms : samples_ms)
    {
        total_ms += sample_ms;
    }

    BenchmarkStats stats;
    stats.avg_ms = total_ms / static_cast<double>(samples_ms.size());
    stats.p50_ms = percentile_ms(samples_ms, 0.50);
    stats.p95_ms = percentile_ms(samples_ms, 0.95);
    return stats;
}

void print_result(const std::string& name, double avg_ms)
{
    std::cout << std::left << std::setw(28) << name
              << " avg=" << std::fixed << std::setprecision(4) << avg_ms << " ms" << std::endl;
}

void print_gemm_result(const std::string& name, const BenchmarkStats& stats, double gflops, double speedup_vs_fp32)
{
    std::cout << std::left << std::setw(20) << name
              << " avg=" << std::fixed << std::setprecision(4) << stats.avg_ms << " ms"
              << " p50=" << stats.p50_ms << " ms"
              << " p95=" << stats.p95_ms << " ms"
              << " throughput=" << std::setprecision(2) << gflops << " GFLOP/s"
              << " speedup=" << std::setprecision(2) << speedup_vs_fp32 << "x"
              << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const BenchmarkOptions opts = parse_args(argc, argv);
        const int active_threads = configured_threads(opts);
        const int head_dim = opts.hidden / opts.head_count;

        std::mt19937 rng(20260306);

        Mat gemm_a({opts.batch, opts.seq_len, opts.hidden}, DT_32F);
        Mat gemm_w({opts.gemm_out, opts.hidden}, DT_32F);
        Mat gemm_w_fp16;
        Mat gemm_w_int8;
        Mat gemm_w_int8_scales;
        fill_random(gemm_a, rng);
        fill_random(gemm_w, rng);
        gemm_w.convertTo(gemm_w_fp16, DT_16F);
        quantize_int8_per_row(gemm_w, gemm_w_int8, gemm_w_int8_scales);

        const BenchmarkStats gemm_fp32_stats = benchmark_stats(opts, [&]() {
            Mat out = gemm(gemm_a, gemm_w, false, true);
            (void)out;
        });

        const BenchmarkStats gemm_fp16_stats = benchmark_stats(opts, [&]() {
            Mat out = gemm(gemm_a, gemm_w_fp16, false, true);
            (void)out;
        });

        const BenchmarkStats gemm_int8_stats = benchmark_stats(opts, [&]() {
            Mat out = gemm(gemm_a, gemm_w_int8, gemm_w_int8_scales, false, true);
            (void)out;
        });

        const double gemm_flops = 2.0 * static_cast<double>(opts.batch) * static_cast<double>(opts.seq_len) *
                                  static_cast<double>(opts.gemm_out) * static_cast<double>(opts.hidden);
        const double gemm_fp32_gflops = gemm_flops / (gemm_fp32_stats.avg_ms * 1e6);
        const double gemm_fp16_gflops = gemm_flops / (gemm_fp16_stats.avg_ms * 1e6);
        const double gemm_int8_gflops = gemm_flops / (gemm_int8_stats.avg_ms * 1e6);

        if (opts.gemm_only)
        {
            std::cout << "minfer gemm benchmark" << std::endl;
            std::cout << "threads=" << active_threads << std::endl;
            std::cout << "shape(batch, seq, hidden, out)=(" << opts.batch << ", " << opts.seq_len
                      << ", " << opts.hidden << ", " << opts.gemm_out << ")" << std::endl;
            print_gemm_result("gemm nt fp32", gemm_fp32_stats, gemm_fp32_gflops, 1.0);
            print_gemm_result("gemm nt fp16", gemm_fp16_stats, gemm_fp16_gflops,
                              gemm_fp32_stats.avg_ms / gemm_fp16_stats.avg_ms);
            print_gemm_result("gemm nt int8", gemm_int8_stats, gemm_int8_gflops,
                              gemm_fp32_stats.avg_ms / gemm_int8_stats.avg_ms);
            return 0;
        }

        Mat add_a({opts.batch, opts.seq_len, opts.hidden}, DT_32F);
        Mat add_b({opts.batch, opts.seq_len, opts.hidden}, DT_32F);
        Mat add_out;
        fill_random(add_a, rng);
        fill_random(add_b, rng);

        Mat row_vec({opts.hidden}, DT_32F);
        Mat row_mul_out;
        fill_random(row_vec, rng);

        Mat trans_in({opts.batch, opts.seq_len, opts.hidden}, DT_32F);
        fill_random(trans_in, rng);

        Mat rope_q_base({opts.seq_len, opts.head_count, head_dim}, DT_32F);
        Mat rope_k_base({opts.seq_len, opts.head_count_kv, head_dim}, DT_32F);
        Mat rope_q_work;
        Mat rope_k_work;
        fill_random(rope_q_base, rng);
        fill_random(rope_k_base, rng);

        Mat softmax_in({opts.batch, opts.seq_len, opts.head_count, head_dim}, DT_32F);
        Mat softmax_out;
        fill_random(softmax_in, rng);

        Mat silu_in({opts.batch, opts.seq_len, opts.hidden}, DT_32F);
        Mat silu_out;
        fill_random(silu_in, rng);

        Mat rmsnorm_in({opts.batch, opts.seq_len, opts.hidden}, DT_32F);
        Mat rmsnorm_w({opts.hidden}, DT_32F);
        Mat rmsnorm_out;
        fill_random(rmsnorm_in, rng);
        fill_random(rmsnorm_w, rng);

        const double add_ms = benchmark_ms(opts, [&]() {
            add(add_a, add_b, add_out);
        });

        const double mul_row_ms = benchmark_ms(opts, [&]() {
            multiply(row_vec, add_b, row_mul_out);
        });

        const double transpose_ms = benchmark_ms(opts, [&]() {
            Mat out = transpose(trans_in);
            (void)out;
        });

        const double rope_ms = benchmark_ms(opts, [&]() {
            rope_q_base.copyTo(rope_q_work);
            rope_k_base.copyTo(rope_k_work);
            rope(rope_q_work, rope_k_work, 0);
        });

        const double softmax_ms = benchmark_ms(opts, [&]() {
            softmax(softmax_in, softmax_out);
        });

        const double silu_ms = benchmark_ms(opts, [&]() {
            silu(silu_in, silu_out);
        });

        const double rmsnorm_ms = benchmark_ms(opts, [&]() {
            rmsnorm(rmsnorm_in, rmsnorm_w, rmsnorm_out, 1e-6f);
        });

        std::cout << "minfer operator benchmark" << std::endl;
        std::cout << "threads=" << active_threads << std::endl;
        std::cout << "shape(batch, seq, hidden)=(" << opts.batch << ", " << opts.seq_len << ", " << opts.hidden << ")" << std::endl;
        std::cout << "gemm(batch, seq, hidden, out)=(" << opts.batch << ", " << opts.seq_len << ", "
                  << opts.hidden << ", " << opts.gemm_out << ")" << std::endl;
        std::cout << "rope(seq, q_heads, kv_heads, head_dim)=(" << opts.seq_len << ", "
                  << opts.head_count << ", " << opts.head_count_kv << ", " << head_dim << ")" << std::endl;

        print_gemm_result("gemm nt fp32", gemm_fp32_stats, gemm_fp32_gflops, 1.0);
        print_gemm_result("gemm nt fp16", gemm_fp16_stats, gemm_fp16_gflops,
                          gemm_fp32_stats.avg_ms / gemm_fp16_stats.avg_ms);
        print_gemm_result("gemm nt int8", gemm_int8_stats, gemm_int8_gflops,
                          gemm_fp32_stats.avg_ms / gemm_int8_stats.avg_ms);
        print_result("add same-shape", add_ms);
        print_result("mul row-broadcast", mul_row_ms);
        print_result("transpose last-two", transpose_ms);
        print_result("rope inplace", rope_ms);
        print_result("softmax lastdim", softmax_ms);
        print_result("silu elementwise", silu_ms);
        print_result("rmsnorm lastdim", rmsnorm_ms);

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "operator benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
