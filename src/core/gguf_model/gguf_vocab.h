//
// Created by mzh on 9/10/2025.
//

#ifndef MINFER_GGUF_VOCAB_H
#define MINFER_GGUF_VOCAB_H

#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include "minfer/net.h"
#include "gguf_loader.h"

namespace minfer {
struct LLama_loader;

typedef int32_t llama_pos;
typedef int32_t llama_token;
typedef int32_t llama_seq_id;
#define LLAMA_TOKEN_NULL -1

typedef enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
} llama_token_type;

typedef enum llama_vocab_type {
    LLAMA_VOCAB_TYPE_NONE = 0, // For models without vocab
    LLAMA_VOCAB_TYPE_SPM  = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  = 2, // GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  = 3, // BERT tokenizer based on WordPiece
} llama_vocab_type;

// pre-tokenization types
typedef enum llama_vocab_pre_type {
    LLAMA_VOCAB_PRE_TYPE_DEFAULT        = 0,
    LLAMA_VOCAB_PRE_TYPE_LLAMA3         = 1,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM   = 2,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3,
    LLAMA_VOCAB_PRE_TYPE_FALCON         = 4,
    LLAMA_VOCAB_PRE_TYPE_MPT            = 5,
    LLAMA_VOCAB_PRE_TYPE_STARCODER      = 6,
    LLAMA_VOCAB_PRE_TYPE_GPT2           = 7,
    LLAMA_VOCAB_PRE_TYPE_REFACT         = 8,
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R      = 9,
    LLAMA_VOCAB_PRE_TYPE_STABLELM2      = 10,
    LLAMA_VOCAB_PRE_TYPE_QWEN2          = 11,
    LLAMA_VOCAB_PRE_TYPE_OLMO           = 12,
    LLAMA_VOCAB_PRE_TYPE_DBRX           = 13,
    LLAMA_VOCAB_PRE_TYPE_SMAUG          = 14,
    LLAMA_VOCAB_PRE_TYPE_PORO           = 15,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3       = 16,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4       = 17,
    LLAMA_VOCAB_PRE_TYPE_VIKING         = 18,
    LLAMA_VOCAB_PRE_TYPE_JAIS           = 19,
    LLAMA_VOCAB_PRE_TYPE_TEKKEN         = 20,
    LLAMA_VOCAB_PRE_TYPE_SMOLLM         = 21,
    LLAMA_VOCAB_PRE_TYPE_CODESHELL      = 22,
    LLAMA_VOCAB_PRE_TYPE_BLOOM          = 23,
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH   = 24,
    LLAMA_VOCAB_PRE_TYPE_EXAONE         = 25,
    LLAMA_VOCAB_PRE_TYPE_CHAMELEON      = 26,
    LLAMA_VOCAB_PRE_TYPE_MINERVA        = 27,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM  = 28,
} llama_vocab_pre_type;

class GGUF_Vocab
{
public:
    GGUF_Vocab();
    ~GGUF_Vocab();

    bool loadFromGGUF(const minfer::LLama_loader& ggufModel);
    //
    // int32_t getTokenId(const std::string& token) const;
    // std::string getTokenString(int32_t tokenId) const;
    //
    //
    // size_t size() const { return tokenToIdMap.size(); }

    // 调用接口
    void decode(const std::vector<int> &out_ids, std::string &out_text);

    void encode(const std::string text, std::vector<int> &out_ids);

private:
    uint32_t n_token_types = 0; // for BERT-style token types

    llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
    llama_vocab_pre_type pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

    llama_token special_bos_id  = 1;
    llama_token special_eos_id  = 2;
    llama_token special_eot_id  = LLAMA_TOKEN_NULL;
    llama_token special_eom_id  = LLAMA_TOKEN_NULL;
    llama_token special_unk_id  = 0;
    llama_token special_sep_id  = LLAMA_TOKEN_NULL;
    llama_token special_pad_id  = LLAMA_TOKEN_NULL;
    llama_token special_mask_id = LLAMA_TOKEN_NULL;
};
}

#endif // MINFER_GGUF_VOCAB_H