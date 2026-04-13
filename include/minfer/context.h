//
// Created by Antigravity on 2026/3/3.
//

#ifndef MINFER_CONTEXT_H
#define MINFER_CONTEXT_H

#include <limits>
#include <vector>

namespace minfer
{

class BenchmarkProfiler;

enum class RuntimePrecision {
    FP32,
    FP16,
    INT8,
};

enum class InferPhase {
    Prefill,   // 首次处理完整 prompt
    Decode     // 逐 token 自回归生成
};

enum class DecodeOutputMode {
    FullLogits,
    ArgMax,
    TopK,
};

struct DecodeCandidate
{
    int token_id = -1;
    float logit = -std::numeric_limits<float>::infinity();
};

struct DecodeSelection
{
    bool ready = false;
    int token_id = -1;
    float logit = -std::numeric_limits<float>::infinity();
    std::vector<DecodeCandidate> top_k;

    void reset(DecodeOutputMode mode, int requested_top_k)
    {
        ready = false;
        token_id = -1;
        logit = -std::numeric_limits<float>::infinity();
        top_k.clear();
        if (mode == DecodeOutputMode::TopK && requested_top_k > 0)
        {
            top_k.reserve(requested_top_k);
        }
    }
};

struct InferenceContext {
    InferPhase phase = InferPhase::Prefill;
    int start_pos = 0;   // 当前序列在全局中的起始位置
    int seq_len   = 0;   // 本次输入的 token 数（prefill=N, decode=1）
    DecodeOutputMode decode_output_mode = DecodeOutputMode::FullLogits;
    int top_k = 0;
    DecodeSelection* decode_selection = nullptr;
    BenchmarkProfiler* benchmark_profiler = nullptr;
    int benchmark_layer_id = -1;
};

}

#endif //MINFER_CONTEXT_H
