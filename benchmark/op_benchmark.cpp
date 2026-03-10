#include "minfer.h"
#include "backend/cpu/kernel/gemm_kernel_xsimd.h"
#include "backend/cpu/layer/runtime_weight.h"

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
    bool gemm_micro_only = false;
    bool gemm_runtime_only = false;
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
        else if (arg == "--gemm-micro-only")
            opts.gemm_micro_only = true;
        else if (arg == "--gemm-runtime-only")
            opts.gemm_runtime_only = true;
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
                << "  --gemm-only\n"
                << "  --gemm-micro-only\n"
                << "  --gemm-runtime-only\n";
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
    const int gemm_mode_count = static_cast<int>(opts.gemm_only) +
                                static_cast<int>(opts.gemm_micro_only) +
                                static_cast<int>(opts.gemm_runtime_only);
    if (gemm_mode_count > 1)
    {
        throw std::runtime_error("Only one of --gemm-only, --gemm-micro-only, --gemm-runtime-only may be set");
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

Mat transpose_last_two_2d(const Mat& input)
{
    return transposeND(input, {1, 0});
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

        RuntimeWeight rw_fp32;
        RuntimeWeight rw_fp16;
        RuntimeWeight rw_int8;
        rw_fp32.init(gemm_w, Int8QuantScheme::PerRow, true, RuntimePrecision::FP32);
        rw_fp16.init(gemm_w, Int8QuantScheme::PerRow, true, RuntimePrecision::FP16);
        rw_int8.init(gemm_w, Int8QuantScheme::PerRow, true, RuntimePrecision::INT8);

        if (opts.gemm_micro_only)
        {
            const int rows = opts.batch * opts.seq_len;
            const double kernel_flops = 2.0 * static_cast<double>(rows) * static_cast<double>(opts.gemm_out) *
                                        static_cast<double>(opts.hidden);

            Mat gemm_w_kn = transpose_last_two_2d(gemm_w);
            Mat gemm_w_kn_fp16 = transpose_last_two_2d(gemm_w_fp16);
            Mat gemm_w_kn_int8 = transpose_last_two_2d(gemm_w_int8);

            std::vector<float> packed_b_fp32(cpu::gemm_xsimd_packed_b_elements(opts.gemm_out, opts.hidden));
            std::vector<hfloat> packed_b_fp16(cpu::gemm_xsimd_packed_b_elements(opts.gemm_out, opts.hidden));
            std::vector<int8_t> packed_b_int8(cpu::gemm_xsimd_packed_b_elements(opts.gemm_out, opts.hidden));
            std::vector<float> packed_scales(cpu::gemm_xsimd_packed_scale_elements(opts.gemm_out));
            std::vector<float> out_nt(static_cast<size_t>(rows) * static_cast<size_t>(opts.gemm_out));
            std::vector<float> out_row_packed(static_cast<size_t>(rows) * static_cast<size_t>(opts.gemm_out));

            cpu::gemm_pack_xsimd_nn_fp32(reinterpret_cast<const float*>(gemm_w_kn.data),
                                         packed_b_fp32.data(),
                                         opts.gemm_out,
                                         opts.hidden);
            cpu::gemm_pack_xsimd_nn_fp16(reinterpret_cast<const hfloat*>(gemm_w_kn_fp16.data),
                                         packed_b_fp16.data(),
                                         opts.gemm_out,
                                         opts.hidden);
            cpu::gemm_pack_xsimd_nn_i8_rowwise(reinterpret_cast<const int8_t*>(gemm_w_kn_int8.data),
                                               reinterpret_cast<const float*>(gemm_w_int8_scales.data),
                                               packed_b_int8.data(),
                                               packed_scales.data(),
                                               opts.gemm_out,
                                               opts.hidden);

            const float* a_ptr = reinterpret_cast<const float*>(gemm_a.data);
            const float* w_fp32_ptr = reinterpret_cast<const float*>(gemm_w.data);
            const hfloat* w_fp16_ptr = reinterpret_cast<const hfloat*>(gemm_w_fp16.data);
            const int8_t* w_int8_ptr = reinterpret_cast<const int8_t*>(gemm_w_int8.data);
            const float* w_int8_scales_ptr = reinterpret_cast<const float*>(gemm_w_int8_scales.data);

            const BenchmarkStats nt_fp32_stats = benchmark_stats(opts, [&]() {
                cpu::gemm_kernel_xsimd_nt(a_ptr, w_fp32_ptr, out_nt.data(), rows, opts.gemm_out, opts.hidden);
            });
            const BenchmarkStats nt_fp16_stats = benchmark_stats(opts, [&]() {
                cpu::gemm_kernel_xsimd_nt_fp16(a_ptr, w_fp16_ptr, out_nt.data(), rows, opts.gemm_out, opts.hidden);
            });
            const BenchmarkStats nt_int8_stats = benchmark_stats(opts, [&]() {
                cpu::gemm_kernel_xsimd_nt_i8_rowwise(a_ptr, w_int8_ptr, w_int8_scales_ptr, out_nt.data(), rows, opts.gemm_out, opts.hidden);
            });

            const BenchmarkStats row_packed_fp32_stats = benchmark_stats(opts, [&]() {
                for (int row = 0; row < rows; ++row)
                {
                    cpu::gemm_kernel_xsimd_row_packed_fp32(a_ptr + static_cast<size_t>(row) * opts.hidden,
                                                           packed_b_fp32.data(),
                                                           out_row_packed.data() + static_cast<size_t>(row) * opts.gemm_out,
                                                           opts.gemm_out,
                                                           opts.hidden);
                }
            });
            const BenchmarkStats row_packed_fp16_stats = benchmark_stats(opts, [&]() {
                for (int row = 0; row < rows; ++row)
                {
                    cpu::gemm_kernel_xsimd_row_packed_fp16(a_ptr + static_cast<size_t>(row) * opts.hidden,
                                                           packed_b_fp16.data(),
                                                           out_row_packed.data() + static_cast<size_t>(row) * opts.gemm_out,
                                                           opts.gemm_out,
                                                           opts.hidden);
                }
            });
            const BenchmarkStats row_packed_int8_stats = benchmark_stats(opts, [&]() {
                for (int row = 0; row < rows; ++row)
                {
                    cpu::gemm_kernel_xsimd_row_packed_i8_rowwise(a_ptr + static_cast<size_t>(row) * opts.hidden,
                                                                 packed_b_int8.data(),
                                                                 packed_scales.data(),
                                                                 out_row_packed.data() + static_cast<size_t>(row) * opts.gemm_out,
                                                                 opts.gemm_out,
                                                                 opts.hidden);
                }
            });

            const double nt_fp32_gflops = kernel_flops / (nt_fp32_stats.avg_ms * 1e6);
            const double nt_fp16_gflops = kernel_flops / (nt_fp16_stats.avg_ms * 1e6);
            const double nt_int8_gflops = kernel_flops / (nt_int8_stats.avg_ms * 1e6);
            const double row_packed_fp32_gflops = kernel_flops / (row_packed_fp32_stats.avg_ms * 1e6);
            const double row_packed_fp16_gflops = kernel_flops / (row_packed_fp16_stats.avg_ms * 1e6);
            const double row_packed_int8_gflops = kernel_flops / (row_packed_int8_stats.avg_ms * 1e6);

            std::cout << "minfer gemm micro-kernel benchmark" << std::endl;
            std::cout << "threads=" << active_threads << std::endl;
            std::cout << "rows=" << rows << ", hidden=" << opts.hidden << ", out=" << opts.gemm_out << std::endl;
            std::cout << "nt entry uses the simple path when rows <= 1 or blocked heuristics stay disabled" << std::endl;
            print_gemm_result("nt entry fp32", nt_fp32_stats, nt_fp32_gflops, 1.0);
            print_gemm_result("nt entry fp16", nt_fp16_stats, nt_fp16_gflops, nt_fp32_stats.avg_ms / nt_fp16_stats.avg_ms);
            print_gemm_result("nt entry int8", nt_int8_stats, nt_int8_gflops, nt_fp32_stats.avg_ms / nt_int8_stats.avg_ms);
            print_gemm_result("rowpacked fp32", row_packed_fp32_stats, row_packed_fp32_gflops, 1.0);
            print_gemm_result("rowpacked fp16", row_packed_fp16_stats, row_packed_fp16_gflops,
                              row_packed_fp32_stats.avg_ms / row_packed_fp16_stats.avg_ms);
            print_gemm_result("rowpacked int8", row_packed_int8_stats, row_packed_int8_gflops,
                              row_packed_fp32_stats.avg_ms / row_packed_int8_stats.avg_ms);
            return 0;
        }

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

        if (opts.gemm_runtime_only)
        {
            const BenchmarkStats runtime_fp32_stats = benchmark_stats(opts, [&]() {
                Mat out = rw_fp32.gemmNT(gemm_a);
                (void)out;
            });

            const BenchmarkStats runtime_fp16_stats = benchmark_stats(opts, [&]() {
                Mat out = rw_fp16.gemmNT(gemm_a);
                (void)out;
            });

            const BenchmarkStats runtime_int8_stats = benchmark_stats(opts, [&]() {
                Mat out = rw_int8.gemmNT(gemm_a);
                (void)out;
            });

            const double runtime_fp32_gflops = gemm_flops / (runtime_fp32_stats.avg_ms * 1e6);
            const double runtime_fp16_gflops = gemm_flops / (runtime_fp16_stats.avg_ms * 1e6);
            const double runtime_int8_gflops = gemm_flops / (runtime_int8_stats.avg_ms * 1e6);

            std::cout << "minfer runtime-weight gemm benchmark" << std::endl;
            std::cout << "threads=" << active_threads << std::endl;
            std::cout << "shape(batch, seq, hidden, out)=(" << opts.batch << ", " << opts.seq_len
                      << ", " << opts.hidden << ", " << opts.gemm_out << ")" << std::endl;
            std::cout << "decode_packed_eligible=" << (rw_fp32.shouldUseDecodePacked(gemm_a) ? "true" : "false") << std::endl;
            print_gemm_result("runtime fp32", runtime_fp32_stats, runtime_fp32_gflops, 1.0);
            print_gemm_result("runtime fp16", runtime_fp16_stats, runtime_fp16_gflops,
                              runtime_fp32_stats.avg_ms / runtime_fp16_stats.avg_ms);
            print_gemm_result("runtime int8", runtime_int8_stats, runtime_int8_gflops,
                              runtime_fp32_stats.avg_ms / runtime_int8_stats.avg_ms);
            return 0;
        }

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
