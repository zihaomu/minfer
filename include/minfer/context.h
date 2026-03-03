//
// Created by Antigravity on 2026/3/3.
//

#ifndef MINFER_CONTEXT_H
#define MINFER_CONTEXT_H

namespace minfer
{

enum class InferPhase {
    Prefill,   // 首次处理完整 prompt
    Decode     // 逐 token 自回归生成
};

struct InferenceContext {
    InferPhase phase = InferPhase::Prefill;
    int start_pos = 0;   // 当前序列在全局中的起始位置
    int seq_len   = 0;   // 本次输入的 token 数（prefill=N, decode=1）
};

}

#endif //MINFER_CONTEXT_H
