#include "minfer.h"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

using namespace minfer;

namespace {

struct BenchmarkOptions
{
    int warmup = 10;
    int iters = 100;
    int batch = 4;
    int seq_len = 128;
    int hidden = 1024;
    int head_count = 8;
    int head_count_kv = 4;
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
        else if (arg == "--head-count")
            opts.head_count = std::stoi(need_value(arg));
        else if (arg == "--head-count-kv")
            opts.head_count_kv = std::stoi(need_value(arg));
        else if (arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [options]\n"
                << "  --warmup <n>\n"
                << "  --iters <n>\n"
                << "  --batch <n>\n"
                << "  --seq-len <n>\n"
                << "  --hidden <n>\n"
                << "  --head-count <n>\n"
                << "  --head-count-kv <n>\n";
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
    if (opts.hidden % opts.head_count != 0)
    {
        throw std::runtime_error("--hidden must be divisible by --head-count");
    }

    return opts;
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

void print_result(const std::string& name, double avg_ms)
{
    std::cout << std::left << std::setw(28) << name
              << " avg=" << std::fixed << std::setprecision(4) << avg_ms << " ms" << std::endl;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        const BenchmarkOptions opts = parse_args(argc, argv);
        const int head_dim = opts.hidden / opts.head_count;

        std::mt19937 rng(20260306);

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
        std::cout << "shape(batch, seq, hidden)=(" << opts.batch << ", " << opts.seq_len << ", " << opts.hidden << ")" << std::endl;
        std::cout << "rope(seq, q_heads, kv_heads, head_dim)=(" << opts.seq_len << ", "
                  << opts.head_count << ", " << opts.head_count_kv << ", " << head_dim << ")" << std::endl;

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
