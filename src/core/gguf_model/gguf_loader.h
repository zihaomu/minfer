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

// quantized type.
enum GGML_TYPE {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    // GGML_TYPE_Q4_0    = 2,
    // GGML_TYPE_Q4_1    = 3,
    // // GGML_TYPE_Q4_2 = 4, support has been removed
    // // GGML_TYPE_Q4_3 = 5, support has been removed
    // GGML_TYPE_Q5_0    = 6,
    // GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 2,
    // GGML_TYPE_Q8_1    = 9,
    // GGML_TYPE_Q2_K    = 10,
    // GGML_TYPE_Q3_K    = 11,
    // GGML_TYPE_Q4_K    = 12,
    // GGML_TYPE_Q5_K    = 13,
    // GGML_TYPE_Q6_K    = 14,
    // GGML_TYPE_Q8_K    = 15,
    // GGML_TYPE_IQ2_XXS = 16,
    // GGML_TYPE_IQ2_XS  = 17,
    // GGML_TYPE_IQ3_XXS = 18,
    // GGML_TYPE_IQ1_S   = 19,
    // GGML_TYPE_IQ4_NL  = 20,
    // GGML_TYPE_IQ3_S   = 21,
    // GGML_TYPE_IQ2_S   = 22,
    // GGML_TYPE_IQ4_XS  = 23,
    // GGML_TYPE_I8      = 24,
    // GGML_TYPE_I16     = 25,
    // GGML_TYPE_I32     = 26,
    // GGML_TYPE_I64     = 27,
    // GGML_TYPE_F64     = 28,
    // GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_COUNT,
};

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
    LLM_TENSOR_TOKEN_EMBD = 0,
    LLM_TENSOR_TOKEN_EMBD_NORM = 1,
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

static const std::map<LLM_KV, const char *> LLM_KV_NAMES = {
        { LLM_KV_GENERAL_ARCHITECTURE,          "general.architecture"                  },
        { LLM_KV_GENERAL_QUANTIZATION_VERSION,  "general.quantization_version"          },
        { LLM_KV_GENERAL_ALIGNMENT,             "general.alignment"                     },
        { LLM_KV_GENERAL_NAME,                  "general.name"                          },
        { LLM_KV_GENERAL_AUTHOR,                "general.author"                        },
        { LLM_KV_GENERAL_URL,                   "general.url"                           },
        { LLM_KV_GENERAL_DESCRIPTION,           "general.description"                   },
        { LLM_KV_GENERAL_LICENSE,               "general.license"                       },
        { LLM_KV_GENERAL_SOURCE_URL,            "general.source.url"                    },
        { LLM_KV_GENERAL_SOURCE_HF_REPO,        "general.source.huggingface.repository" },

        { LLM_KV_VOCAB_SIZE,                    "%s.vocab_size"            },
        { LLM_KV_CONTEXT_LENGTH,                "%s.context_length"        },
        { LLM_KV_EMBEDDING_LENGTH,              "%s.embedding_length"      },
        { LLM_KV_BLOCK_COUNT,                   "%s.block_count"           },
        { LLM_KV_FEED_FORWARD_LENGTH,           "%s.feed_forward_length"   },
        { LLM_KV_USE_PARALLEL_RESIDUAL,         "%s.use_parallel_residual" },
        { LLM_KV_TENSOR_DATA_LAYOUT,            "%s.tensor_data_layout"    },
        { LLM_KV_EXPERT_COUNT,                  "%s.expert_count"          },
        { LLM_KV_EXPERT_USED_COUNT,             "%s.expert_used_count"     },
        { LLM_KV_POOLING_TYPE ,                 "%s.pooling_type"          },
        { LLM_KV_LOGIT_SCALE,                   "%s.logit_scale"           },

        { LLM_KV_ATTENTION_HEAD_COUNT,          "%s.attention.head_count"             },
        { LLM_KV_ATTENTION_HEAD_COUNT_KV,       "%s.attention.head_count_kv"          },
        { LLM_KV_ATTENTION_MAX_ALIBI_BIAS,      "%s.attention.max_alibi_bias"         },
        { LLM_KV_ATTENTION_CLAMP_KQV,           "%s.attention.clamp_kqv"              },
        { LLM_KV_ATTENTION_KEY_LENGTH,          "%s.attention.key_length"             },
        { LLM_KV_ATTENTION_VALUE_LENGTH,        "%s.attention.value_length"           },
        { LLM_KV_ATTENTION_LAYERNORM_EPS,       "%s.attention.layer_norm_epsilon"     },
        { LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,   "%s.attention.layer_norm_rms_epsilon" },
        { LLM_KV_ATTENTION_CAUSAL,              "%s.attention.causal"                 },

        { LLM_KV_ROPE_DIMENSION_COUNT,          "%s.rope.dimension_count"                 },
        { LLM_KV_ROPE_FREQ_BASE,                "%s.rope.freq_base"                       },
        { LLM_KV_ROPE_SCALE_LINEAR,             "%s.rope.scale_linear"                    },
        { LLM_KV_ROPE_SCALING_TYPE,             "%s.rope.scaling.type"                    },
        { LLM_KV_ROPE_SCALING_FACTOR,           "%s.rope.scaling.factor"                  },
        { LLM_KV_ROPE_SCALING_ORIG_CTX_LEN,     "%s.rope.scaling.original_context_length" },
        { LLM_KV_ROPE_SCALING_FINETUNED,        "%s.rope.scaling.finetuned"               },

        { LLM_KV_SPLIT_NO,                      "split.no"            },
        { LLM_KV_SPLIT_COUNT,                   "split.count"         },
        { LLM_KV_SPLIT_TENSORS_COUNT,           "split.tensors.count" },

        { LLM_KV_SSM_CONV_KERNEL,               "%s.ssm.conv_kernel"    },
        { LLM_KV_SSM_INNER_SIZE,                "%s.ssm.inner_size"     },
        { LLM_KV_SSM_STATE_SIZE,                "%s.ssm.state_size"     },
        { LLM_KV_SSM_TIME_STEP_RANK,            "%s.ssm.time_step_rank" },

        { LLM_KV_TOKENIZER_MODEL,               "tokenizer.ggml.model"              },
        { LLM_KV_TOKENIZER_LIST,                "tokenizer.ggml.tokens"             },
        { LLM_KV_TOKENIZER_TOKEN_TYPE,          "tokenizer.ggml.token_type"         },
        { LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,    "tokenizer.ggml.token_type_count"   },
        { LLM_KV_TOKENIZER_SCORES,              "tokenizer.ggml.scores"             },
        { LLM_KV_TOKENIZER_MERGES,              "tokenizer.ggml.merges"             },
        { LLM_KV_TOKENIZER_BOS_ID,              "tokenizer.ggml.bos_token_id"       },
        { LLM_KV_TOKENIZER_EOS_ID,              "tokenizer.ggml.eos_token_id"       },
        { LLM_KV_TOKENIZER_UNK_ID,              "tokenizer.ggml.unknown_token_id"   },
        { LLM_KV_TOKENIZER_SEP_ID,              "tokenizer.ggml.seperator_token_id" },
        { LLM_KV_TOKENIZER_PAD_ID,              "tokenizer.ggml.padding_token_id"   },
        { LLM_KV_TOKENIZER_ADD_BOS,             "tokenizer.ggml.add_bos_token"      },
        { LLM_KV_TOKENIZER_ADD_EOS,             "tokenizer.ggml.add_eos_token"      },
        { LLM_KV_TOKENIZER_ADD_PREFIX,          "tokenizer.ggml.add_space_prefix"   },
        { LLM_KV_TOKENIZER_HF_JSON,             "tokenizer.huggingface.json"        },
        { LLM_KV_TOKENIZER_RWKV,                "tokenizer.rwkv.world"              },
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


// note: these values should be synchronized with ggml_rope
// TODO: maybe move this enum to ggml.h (ggml_rope_type)
enum llama_rope_type {
    LLAMA_ROPE_TYPE_NONE = -1,
    LLAMA_ROPE_TYPE_NORM =  0,
    LLAMA_ROPE_TYPE_NEOX =  2,
    LLAMA_ROPE_TYPE_GLM  =  4,
};

// model file types
enum llama_ftype {
    LLAMA_FTYPE_ALL_F32              = 0,
    LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
    // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
    // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
    LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors

    LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
};

enum llama_rope_scaling_type {
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_YARN,
};

enum llama_pooling_type {
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
    LLAMA_POOLING_TYPE_NONE = 0,
    LLAMA_POOLING_TYPE_MEAN = 1,
    LLAMA_POOLING_TYPE_CLS  = 2,
};

enum llama_split_mode {
    LLAMA_SPLIT_MODE_NONE    = 0, // single GPU
    LLAMA_SPLIT_MODE_LAYER   = 1, // split layers and KV across GPUs
    LLAMA_SPLIT_MODE_ROW     = 2, // split rows across GPUs
};

class  GGUF_Vocab;

/// 创建gguf net params
/// \param path gguf 模型路径
/// \param allLayerParams
void readGGUF(const std::string path, std::vector<std::shared_ptr<LayerParams> >& netParams, std::shared_ptr<GGUF_Vocab>& gguf_vocab);

}
#endif //MINFER_GGUF_LOADER_H
