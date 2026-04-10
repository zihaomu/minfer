// Chat example based on Minfer with prefill/decode and KV cache

#include "minfer.h"

using namespace minfer;

int main()
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf", RuntimePrecision::FP32);

    // Tokenizer 编码
    std::string prompt = "Who are";
    std::vector<int> prompt_ids;
    net.encode(prompt, prompt_ids);

    // Prefill: 处理完整 prompt
    DecodeResult prefill_result = net.prefillDecode(prompt_ids, DecodeOutputMode::ArgMax);
    int next_token = prefill_result.token_id;

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generated: ";

    // 先输出 prefill 阶段得到的第一个生成 token（避免漏掉第一个 token）
    std::string token_text;
    net.decode({next_token}, token_text);
    std::cout << token_text << std::flush;

    // Autoregressive Decode Loop
    int max_new_tokens = 50;
    for (int i = 1; i < max_new_tokens; i++)
    {
        DecodeResult step_result = net.stepDecode(next_token, DecodeOutputMode::ArgMax);
        next_token = step_result.token_id;

        // Tokenizer 解码
        token_text.clear();
        net.decode({next_token}, token_text);
        std::cout <<token_text << std::flush;

        // TODO: 检查 EOS token 以提前终止
        // if (next_token == eos_token_id) break;
    }
    std::cout << std::endl;

    // 重置 KV Cache 以开始新一轮对话
    net.resetKVCache();

    return 0;
}
