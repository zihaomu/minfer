//
// Created by mzh on 9/10/2025.
//

#include "gguf_vocab.h"
#include "minfer.h"

namespace minfer {
struct GGUF_context;

GGUF_Vocab::GGUF_Vocab() 
{
}

GGUF_Vocab::~GGUF_Vocab() 
{

}

bool GGUF_Vocab::loadFromGGUF(const struct minfer::LLama_loader& loader)
{
    std::shared_ptr<GGUF_context> ctx = loader.meta;
    if (!ctx) 
    {
        M_Error(NULL, "GGUF context is null");
        return false;
    }

    // determine the vocabulary type
    {
        std::string tokenizer_model;
        std::string tokenizer_pre;

        loader.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);
        loader.get_key(LLM_KV_TOKENIZER_PRE,   tokenizer_pre, false);
        loader.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, n_token_types, false);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") 
        {
            type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            special_bos_id  = LLAMA_TOKEN_NULL;
            special_eos_id  = LLAMA_TOKEN_NULL;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
            linefeed_id     = LLAMA_TOKEN_NULL;

            // read vocab size from metadata
            uint32_t n_tokens = 0;
            if (loader.get_key(LLM_KV_VOCAB_SIZE, n_tokens, false)) 
            {
                DEBUG_PRINT("%s: adding %u dummy tokens\n", __func__, n_tokens);

                id_to_token.resize(n_tokens);
            }

            return true;
        }

        if (tokenizer_model == "llama") {
            type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            special_bos_id  = 1;
            special_eos_id  = 2;
            special_unk_id  = 0;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
        }
        else
        {
            M_Error_(NULL, ("Unsupported tokenizer model: %s", tokenizer_model.c_str()));
        }

        if (type == LLAMA_VOCAB_TYPE_BPE)
        {
            M_Warning("%s: missing pre-tokenizer type, using: 'default'\n", __func__);
            M_Warning("%s:                                             \n", __func__);
            M_Warning("%s: ************************************        \n", __func__);
            M_Warning("%s: GENERATION QUALITY WILL BE DEGRADED!        \n", __func__);
            M_Warning("%s: CONSIDER REGENERATING THE MODEL             \n", __func__);
            M_Warning("%s: ************************************        \n", __func__);
            M_Warning("%s:                                             \n", __func__);
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        }
    }

    // Load vocabulary from GGUF context
    return true;
}

void GGUF_Vocab::decode(const std::vector<int> &out_ids, std::string &out_text)
{
    // Implement the decoding logic here
}

void GGUF_Vocab::encode(const std::string text, std::vector<int> &out_ids)
{
    // Implement the encoding logic here
}

}

