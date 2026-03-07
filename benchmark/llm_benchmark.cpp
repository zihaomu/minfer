#include "minfer.h"
#include "benchmark_threads.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace minfer;

namespace {

struct BenchmarkOptions {
    std::string model_path = std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf";
    std::string prompt = "Hello world! <s>";
    std::string prompt_file;
    std::vector<int> prompt_lens = {32, 128, 512};
    int decode_tokens = 128;
    int warmup = 1;
    int runs = 5;
    int progress_interval = 32;
    int threads = 0;
    std::string csv_path;
};

struct CaseResult {
    int prompt_len = 0;
    int decode_tokens = 0;

    double prefill_avg_ms = 0.0;
    double prefill_p50_ms = 0.0;
    double prefill_p90_ms = 0.0;
    double prefill_tps = 0.0;

    double decode_avg_ms = 0.0;
    double decode_p50_ms = 0.0;
    double decode_p90_ms = 0.0;
    double decode_p99_ms = 0.0;
    double decode_tps = 0.0;

    double ttft_avg_ms = 0.0;
    double e2e_tps = 0.0;
};

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    if (p <= 0.0) return *std::min_element(values.begin(), values.end());
    if (p >= 100.0) return *std::max_element(values.begin(), values.end());

    std::sort(values.begin(), values.end());
    const double rank = (p / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double w = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - w) + values[hi] * w;
}

std::string trim(const std::string& s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

std::vector<int> parse_csv_ints(const std::string& text) {
    std::vector<int> out;
    std::stringstream ss(text);
    std::string item;

    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) continue;
        int v = std::stoi(item);
        if (v <= 0) {
            throw std::runtime_error("Prompt length must be > 0: " + item);
        }
        out.push_back(v);
    }

    if (out.empty()) {
        throw std::runtime_error("Empty --prompt-lens");
    }
    return out;
}

std::string load_text_file(const std::string& path) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open prompt file: " + path);
    }
    std::ostringstream oss;
    oss << fin.rdbuf();
    return oss.str();
}

void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --model <path>          GGUF model path\n"
        << "  --prompt <text>         Prompt text for tokenizer\n"
        << "  --prompt-file <path>    Read prompt text from file\n"
        << "  --prompt-lens <csv>     Prompt lengths to benchmark, e.g. 32,128,512\n"
        << "  --decode-tokens <n>     Decode tokens per run (default: 128)\n"
        << "  --warmup <n>            Warmup runs (default: 1)\n"
        << "  --runs <n>              Measured runs (default: 5)\n"
        << "  --progress-interval <n> Print decode progress every n tokens (default: 32)\n"
        << "  --threads <n>           Thread count to use (default: all platform threads)\n"
        << "  --csv <path>            Save result table to csv\n"
        << "  --help                  Show help\n";
}

BenchmarkOptions parse_args(int argc, char** argv) {
    BenchmarkOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model") {
            opts.model_path = need_value(arg);
        } else if (arg == "--prompt") {
            opts.prompt = need_value(arg);
        } else if (arg == "--prompt-file") {
            opts.prompt_file = need_value(arg);
        } else if (arg == "--prompt-lens") {
            opts.prompt_lens = parse_csv_ints(need_value(arg));
        } else if (arg == "--decode-tokens") {
            opts.decode_tokens = std::stoi(need_value(arg));
        } else if (arg == "--warmup") {
            opts.warmup = std::stoi(need_value(arg));
        } else if (arg == "--runs") {
            opts.runs = std::stoi(need_value(arg));
        } else if (arg == "--progress-interval") {
            opts.progress_interval = std::stoi(need_value(arg));
        } else if (arg == "--threads") {
            opts.threads = std::stoi(need_value(arg));
        } else if (arg == "--csv") {
            opts.csv_path = need_value(arg);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (!opts.prompt_file.empty()) {
        opts.prompt = load_text_file(opts.prompt_file);
    }

    if (opts.decode_tokens <= 0) {
        throw std::runtime_error("--decode-tokens must be > 0");
    }
    if (opts.warmup < 0) {
        throw std::runtime_error("--warmup must be >= 0");
    }
    if (opts.runs <= 0) {
        throw std::runtime_error("--runs must be > 0");
    }
    if (opts.progress_interval <= 0) {
        throw std::runtime_error("--progress-interval must be > 0");
    }
    if (opts.threads < 0) {
        throw std::runtime_error("--threads must be >= 0");
    }
    return opts;
}

std::vector<int> build_prompt_ids(const std::vector<int>& seed_ids, int prompt_len) {
    M_Assert(prompt_len > 0);
    M_Assert(!seed_ids.empty());

    std::vector<int> out;
    out.reserve(prompt_len);

    while (static_cast<int>(out.size()) < prompt_len) {
        const int remain = prompt_len - static_cast<int>(out.size());
        const int copy_n = std::min(remain, static_cast<int>(seed_ids.size()));
        out.insert(out.end(), seed_ids.begin(), seed_ids.begin() + copy_n);
    }
    return out;
}

int sample_next_token(const Mat& logits) {
    std::vector<int> ids = argmax_tokens(
        reinterpret_cast<const float*>(logits.data),
        logits.size[0], logits.size[1], logits.size[2]);

    M_Assert(!ids.empty());
    return ids.back();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

CaseResult run_case(Net& net, const std::vector<int>& seed_ids, const BenchmarkOptions& opts, int prompt_len) {
    CaseResult res;
    res.prompt_len = prompt_len;
    res.decode_tokens = opts.decode_tokens;

    const std::vector<int> prompt_ids = build_prompt_ids(seed_ids, prompt_len);

    std::vector<double> prefill_runs_ms;
    std::vector<double> decode_runs_ms;
    std::vector<double> decode_step_ms;
    std::vector<double> ttft_runs_ms;

    auto run_once = [&](bool collect, int run_idx, int run_total, const char* phase_name) {
        std::cout << "  [" << phase_name << " " << run_idx << "/" << run_total << "] start" << std::endl;
        net.resetKVCache();

        const double t0 = now_ms();
        Mat logits = net.prefill(prompt_ids);
        const double t1 = now_ms();
        const double prefill_ms = t1 - t0;
        std::cout << "  [" << phase_name << " " << run_idx << "/" << run_total
                  << "] prefill done: " << std::fixed << std::setprecision(2)
                  << prefill_ms << " ms" << std::endl;

        int next_token = sample_next_token(logits);
        double decode_total_ms = 0.0;
        double first_decode_ms = 0.0;

        for (int i = 0; i < opts.decode_tokens; ++i) {
            const double d0 = now_ms();
            logits = net.step(next_token);
            const double d1 = now_ms();

            const double step_ms = d1 - d0;
            decode_total_ms += step_ms;
            if (i == 0) first_decode_ms = step_ms;
            if (collect) decode_step_ms.push_back(step_ms);

            next_token = sample_next_token(logits);

            const int decoded = i + 1;
            if (decoded % opts.progress_interval == 0 || decoded == opts.decode_tokens) {
                std::cout << "  [" << phase_name << " " << run_idx << "/" << run_total
                          << "] decode " << decoded << "/" << opts.decode_tokens
                          << std::endl;
            }
        }

        std::cout << "  [" << phase_name << " " << run_idx << "/" << run_total
                  << "] done: decode_total=" << std::fixed << std::setprecision(2)
                  << decode_total_ms << " ms" << std::endl;

        if (collect) {
            prefill_runs_ms.push_back(prefill_ms);
            decode_runs_ms.push_back(decode_total_ms);
            ttft_runs_ms.push_back(prefill_ms + first_decode_ms);
        }
    };

    for (int i = 0; i < opts.warmup; ++i) {
        run_once(false, i + 1, opts.warmup, "warmup");
    }
    for (int i = 0; i < opts.runs; ++i) {
        run_once(true, i + 1, opts.runs, "run");
    }

    res.prefill_avg_ms = mean(prefill_runs_ms);
    res.prefill_p50_ms = percentile(prefill_runs_ms, 50.0);
    res.prefill_p90_ms = percentile(prefill_runs_ms, 90.0);

    res.decode_avg_ms = mean(decode_step_ms);
    res.decode_p50_ms = percentile(decode_step_ms, 50.0);
    res.decode_p90_ms = percentile(decode_step_ms, 90.0);
    res.decode_p99_ms = percentile(decode_step_ms, 99.0);

    res.ttft_avg_ms = mean(ttft_runs_ms);

    if (res.prefill_avg_ms > 0.0) {
        res.prefill_tps = static_cast<double>(prompt_len) * 1000.0 / res.prefill_avg_ms;
    }
    if (res.decode_avg_ms > 0.0) {
        res.decode_tps = 1000.0 / res.decode_avg_ms;
    }

    const double e2e_avg_ms = mean(prefill_runs_ms) + mean(decode_runs_ms);
    if (e2e_avg_ms > 0.0) {
        res.e2e_tps = static_cast<double>(prompt_len + opts.decode_tokens) * 1000.0 / e2e_avg_ms;
    }

    return res;
}

void print_results(const std::vector<CaseResult>& results) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n=== Benchmark Results ===\n";
    for (const auto& r : results) {
        std::cout << "[prompt_len=" << r.prompt_len << ", decode_tokens=" << r.decode_tokens << "]\n";
        std::cout << "prefill : avg=" << r.prefill_avg_ms
                  << " ms, p50=" << r.prefill_p50_ms
                  << " ms, p90=" << r.prefill_p90_ms
                  << " ms, throughput=" << r.prefill_tps << " tok/s\n";
        std::cout << "decode  : avg=" << r.decode_avg_ms
                  << " ms/tok, p50=" << r.decode_p50_ms
                  << ", p90=" << r.decode_p90_ms
                  << ", p99=" << r.decode_p99_ms
                  << ", throughput=" << r.decode_tps << " tok/s\n";
        std::cout << "ttft    : avg=" << r.ttft_avg_ms << " ms\n";
        std::cout << "end2end : throughput=" << r.e2e_tps << " tok/s (prompt+decode)\n\n";
    }
}

void write_csv(const std::string& path, const std::vector<CaseResult>& results) {
    std::ofstream fout(path);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to write csv: " + path);
    }

    fout << "prompt_len,decode_tokens,prefill_avg_ms,prefill_p50_ms,prefill_p90_ms,prefill_tps,"
         << "decode_avg_ms,decode_p50_ms,decode_p90_ms,decode_p99_ms,decode_tps,ttft_avg_ms,e2e_tps\n";

    fout << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        fout << r.prompt_len << ","
             << r.decode_tokens << ","
             << r.prefill_avg_ms << ","
             << r.prefill_p50_ms << ","
             << r.prefill_p90_ms << ","
             << r.prefill_tps << ","
             << r.decode_avg_ms << ","
             << r.decode_p50_ms << ","
             << r.decode_p90_ms << ","
             << r.decode_p99_ms << ","
             << r.decode_tps << ","
             << r.ttft_avg_ms << ","
             << r.e2e_tps << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        const BenchmarkOptions opts = parse_args(argc, argv);
        const int active_threads = configure_benchmark_threads(opts.threads);

        std::cout << "Model: " << opts.model_path << "\n";
        std::cout << "Prompt lengths: ";
        for (size_t i = 0; i < opts.prompt_lens.size(); ++i) {
            std::cout << opts.prompt_lens[i];
            if (i + 1 != opts.prompt_lens.size()) std::cout << ",";
        }
        std::cout << "\nDecode tokens per run: " << opts.decode_tokens
                  << "\nWarmup: " << opts.warmup
                  << "\nRuns: " << opts.runs
                  << "\nThreads: " << active_threads << "\n";

        Net net;

        const double load_start_ms = now_ms();
        net.readNet(opts.model_path);
        const double load_end_ms = now_ms();
        std::cout << "Model load time: " << std::fixed << std::setprecision(2)
                  << (load_end_ms - load_start_ms) << " ms\n";

        std::vector<int> seed_ids;
        net.encode(opts.prompt, seed_ids);
        if (seed_ids.empty()) {
            throw std::runtime_error("Tokenizer returned empty prompt token ids.");
        }

        std::vector<CaseResult> results;
        results.reserve(opts.prompt_lens.size());

        for (int prompt_len : opts.prompt_lens) {
            std::cout << "Running case prompt_len=" << prompt_len << " ...\n";
            results.push_back(run_case(net, seed_ids, opts, prompt_len));
        }

        print_results(results);

        if (!opts.csv_path.empty()) {
            write_csv(opts.csv_path, results);
            std::cout << "CSV saved to: " << opts.csv_path << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
