//
// Created by mzh on 2024/4/10.
//

#ifndef MINFER_GGUF_LOADER_H
#define MINFER_GGUF_LOADER_H

#include <string>
#include <vector>
#include <map>
#include "minfer/net.h"

namespace minfer
{

enum GGUF_TYPE {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
    GGUF_TYPE_COUNT,       // marks the end of the enum
};

enum LLM_ARCH {
    LLM_ARCH_LLAMA,
    LLM_ARCH_FALCON,
    LLM_ARCH_BAICHUAN,
    LLM_ARCH_GROK,
    LLM_ARCH_GPT2,
    LLM_ARCH_GPTJ,
    LLM_ARCH_GPTNEOX,
    LLM_ARCH_MPT,
    LLM_ARCH_STARCODER,
    LLM_ARCH_PERSIMMON,
    LLM_ARCH_REFACT,
    LLM_ARCH_BERT,
    LLM_ARCH_NOMIC_BERT,
    LLM_ARCH_BLOOM,
    LLM_ARCH_STABLELM,
    LLM_ARCH_QWEN,
    LLM_ARCH_QWEN2,
    LLM_ARCH_PHI2,
    LLM_ARCH_PLAMO,
    LLM_ARCH_CODESHELL,
    LLM_ARCH_ORION,
    LLM_ARCH_INTERNLM2,
    LLM_ARCH_MINICPM,
    LLM_ARCH_GEMMA,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_MAMBA,
    LLM_ARCH_XVERSE,
    LLM_ARCH_COMMAND_R,
    LLM_ARCH_UNKNOWN,
};

static const std::map<LLM_ARCH, const char *> LLM_ARCH_NAMES = {
        { LLM_ARCH_LLAMA,           "llama"      },
        { LLM_ARCH_FALCON,          "falcon"     },
        { LLM_ARCH_GROK,            "grok"       },
        { LLM_ARCH_GPT2,            "gpt2"       },
        { LLM_ARCH_GPTJ,            "gptj"       },
        { LLM_ARCH_GPTNEOX,         "gptneox"    },
        { LLM_ARCH_MPT,             "mpt"        },
        { LLM_ARCH_BAICHUAN,        "baichuan"   },
        { LLM_ARCH_STARCODER,       "starcoder"  },
        { LLM_ARCH_PERSIMMON,       "persimmon"  },
        { LLM_ARCH_REFACT,          "refact"     },
        { LLM_ARCH_BERT,            "bert"       },
        { LLM_ARCH_NOMIC_BERT,      "nomic-bert" },
        { LLM_ARCH_BLOOM,           "bloom"      },
        { LLM_ARCH_STABLELM,        "stablelm"   },
        { LLM_ARCH_QWEN,            "qwen"       },
        { LLM_ARCH_QWEN2,           "qwen2"      },
        { LLM_ARCH_PHI2,            "phi2"       },
        { LLM_ARCH_PLAMO,           "plamo"      },
        { LLM_ARCH_CODESHELL,       "codeshell"  },
        { LLM_ARCH_ORION,           "orion"      },
        { LLM_ARCH_INTERNLM2,       "internlm2"  },
        { LLM_ARCH_MINICPM,         "minicpm"    },
        { LLM_ARCH_GEMMA,           "gemma"      },
        { LLM_ARCH_STARCODER2,      "starcoder2" },
        { LLM_ARCH_MAMBA,           "mamba"      },
        { LLM_ARCH_XVERSE,          "xverse"     },
        { LLM_ARCH_COMMAND_R,       "command-r"  },
        { LLM_ARCH_UNKNOWN,         "(unknown)"  },
};

enum LLM_TENSOR {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_TOKEN_EMBD_NORM,
    LLM_TENSOR_TOKEN_TYPES,
    LLM_TENSOR_POS_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ROPE_FREQS,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_QKV,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_NORM_2,
    LLM_TENSOR_ATTN_OUT_NORM,
    LLM_TENSOR_ATTN_ROT_EMBD,
    LLM_TENSOR_FFN_GATE_INP,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_FFN_ACT,
    LLM_TENSOR_FFN_DOWN_EXP,
    LLM_TENSOR_FFN_GATE_EXP,
    LLM_TENSOR_FFN_UP_EXP,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
    LLM_TENSOR_LAYER_OUT_NORM,
    LLM_TENSOR_SSM_IN,
    LLM_TENSOR_SSM_CONV1D,
    LLM_TENSOR_SSM_X,
    LLM_TENSOR_SSM_DT,
    LLM_TENSOR_SSM_A,
    LLM_TENSOR_SSM_D,
    LLM_TENSOR_SSM_OUT,
};

enum LLM_KV {
    LLM_KV_GENERAL_ARCHITECTURE,
    LLM_KV_GENERAL_QUANTIZATION_VERSION,
    LLM_KV_GENERAL_ALIGNMENT,
    LLM_KV_GENERAL_NAME,
    LLM_KV_GENERAL_AUTHOR,
    LLM_KV_GENERAL_URL,
    LLM_KV_GENERAL_DESCRIPTION,
    LLM_KV_GENERAL_LICENSE,
    LLM_KV_GENERAL_SOURCE_URL,
    LLM_KV_GENERAL_SOURCE_HF_REPO,

    LLM_KV_VOCAB_SIZE,
    LLM_KV_CONTEXT_LENGTH,
    LLM_KV_EMBEDDING_LENGTH,
    LLM_KV_BLOCK_COUNT,
    LLM_KV_FEED_FORWARD_LENGTH,
    LLM_KV_USE_PARALLEL_RESIDUAL,
    LLM_KV_TENSOR_DATA_LAYOUT,
    LLM_KV_EXPERT_COUNT,
    LLM_KV_EXPERT_USED_COUNT,
    LLM_KV_POOLING_TYPE,
    LLM_KV_LOGIT_SCALE,

    LLM_KV_ATTENTION_HEAD_COUNT,
    LLM_KV_ATTENTION_HEAD_COUNT_KV,
    LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
    LLM_KV_ATTENTION_CLAMP_KQV,
    LLM_KV_ATTENTION_KEY_LENGTH,
    LLM_KV_ATTENTION_VALUE_LENGTH,
    LLM_KV_ATTENTION_LAYERNORM_EPS,
    LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
    LLM_KV_ATTENTION_CAUSAL,

    LLM_KV_ROPE_DIMENSION_COUNT,
    LLM_KV_ROPE_FREQ_BASE,
    LLM_KV_ROPE_SCALE_LINEAR,
    LLM_KV_ROPE_SCALING_TYPE,
    LLM_KV_ROPE_SCALING_FACTOR,
    LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,
    LLM_KV_ROPE_SCALING_FINETUNED,

    LLM_KV_SPLIT_NO,
    LLM_KV_SPLIT_COUNT,
    LLM_KV_SPLIT_TENSORS_COUNT,

    LLM_KV_SSM_INNER_SIZE,
    LLM_KV_SSM_CONV_KERNEL,
    LLM_KV_SSM_STATE_SIZE,
    LLM_KV_SSM_TIME_STEP_RANK,

    LLM_KV_TOKENIZER_MODEL,
    LLM_KV_TOKENIZER_LIST,
    LLM_KV_TOKENIZER_TOKEN_TYPE,
    LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
    LLM_KV_TOKENIZER_SCORES,
    LLM_KV_TOKENIZER_MERGES,
    LLM_KV_TOKENIZER_BOS_ID,
    LLM_KV_TOKENIZER_EOS_ID,
    LLM_KV_TOKENIZER_UNK_ID,
    LLM_KV_TOKENIZER_SEP_ID,
    LLM_KV_TOKENIZER_PAD_ID,
    LLM_KV_TOKENIZER_ADD_BOS,
    LLM_KV_TOKENIZER_ADD_EOS,
    LLM_KV_TOKENIZER_ADD_PREFIX,
    LLM_KV_TOKENIZER_HF_JSON,
    LLM_KV_TOKENIZER_RWKV,
};

static const std::map<LLM_ARCH, std::map<LLM_TENSOR, std::string>> LLM_TENSOR_NAMES = {
        {
                LLM_ARCH_LLAMA,
                {
                        {LLM_TENSOR_TOKEN_EMBD, "token_embd"},
                        {LLM_TENSOR_OUTPUT_NORM, "output_norm"},
                        {LLM_TENSOR_OUTPUT, "output"},
                        {LLM_TENSOR_ROPE_FREQS, "rope_freqs"},
                        {LLM_TENSOR_ATTN_NORM, "blk.%d.attn_norm"},
                        {LLM_TENSOR_ATTN_Q, "blk.%d.attn_q"},
                        {LLM_TENSOR_ATTN_K, "blk.%d.attn_k"},
                        {LLM_TENSOR_ATTN_V, "blk.%d.attn_v"},
                        {LLM_TENSOR_ATTN_OUT, "blk.%d.attn_output"},
                        {LLM_TENSOR_ATTN_ROT_EMBD, "blk.%d.attn_rot_embd"},
                        {LLM_TENSOR_FFN_GATE_INP, "blk.%d.ffn_gate_inp"},
                        {LLM_TENSOR_FFN_NORM, "blk.%d.ffn_norm"},
                        {LLM_TENSOR_FFN_GATE, "blk.%d.ffn_gate"},
                        {LLM_TENSOR_FFN_DOWN, "blk.%d.ffn_down"},
                        {LLM_TENSOR_FFN_UP, "blk.%d.ffn_up"},
                        {LLM_TENSOR_FFN_GATE_EXP, "blk.%d.ffn_gate.%d"},
                        {LLM_TENSOR_FFN_DOWN_EXP, "blk.%d.ffn_down.%d"},
                        {LLM_TENSOR_FFN_UP_EXP, "blk.%d.ffn_up.%d"},
                },
        },
        {
                LLM_ARCH_BAICHUAN,
                {
                        {LLM_TENSOR_TOKEN_EMBD, "token_embd"},
                        {LLM_TENSOR_OUTPUT_NORM, "output_norm"},
                        {LLM_TENSOR_OUTPUT, "output"},
                        {LLM_TENSOR_ROPE_FREQS, "rope_freqs"},
                        {LLM_TENSOR_ATTN_NORM, "blk.%d.attn_norm"},
                        {LLM_TENSOR_ATTN_Q, "blk.%d.attn_q"},
                        {LLM_TENSOR_ATTN_K, "blk.%d.attn_k"},
                        {LLM_TENSOR_ATTN_V, "blk.%d.attn_v"},
                        {LLM_TENSOR_ATTN_OUT, "blk.%d.attn_output"},
                        {LLM_TENSOR_ATTN_ROT_EMBD, "blk.%d.attn_rot_embd"},
                        {LLM_TENSOR_FFN_NORM,     "blk.%d.ffn_norm"},
                        {LLM_TENSOR_FFN_GATE, "blk.%d.ffn_gate"},
                        {LLM_TENSOR_FFN_DOWN, "blk.%d.ffn_down"},
                        {LLM_TENSOR_FFN_UP,   "blk.%d.ffn_up"},
                },
        },

        // TODO add more llm model
};

/// 创建gguf net params
/// \param path gguf 模型路径
/// \param allLayerParams
void readGGUF(const std::string path, std::map<int, std::shared_ptr<LayerParams> >& netParams);

}
#endif //MINFER_GGUF_LOADER_H
