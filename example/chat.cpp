// Chat example based on Minfer with prefill/decode and KV cache

#include "minfer.h"

using namespace minfer;

int main()
{
    Net net;
    net.readNet(std::string(M_ROOT_PATH) + "/test/big_models/Lite-Oute-1-65M-FP16.gguf");

    // Tokenizer 编码
    std::string prompt = "Who are";
    std::vector<int> prompt_ids;
    net.encode(prompt, prompt_ids);

    // Prefill: 处理完整 prompt
    Mat logits = net.prefill(prompt_ids);

    // 取最后一个 token 的 argmax 作为生成的第一个 token
    std::vector<int> next_ids = argmax_tokens(
        reinterpret_cast<const float*>(logits.data),
        logits.size[0], logits.size[1], logits.size[2]);
    int next_token = next_ids.back();

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generated: ";

    // Autoregressive Decode Loop
    int max_new_tokens = 50;
    for (int i = 0; i < max_new_tokens; i++)
    {
        logits = net.step(next_token);

        next_ids = argmax_tokens(
            reinterpret_cast<const float*>(logits.data),
            logits.size[0], logits.size[1], logits.size[2]);
        next_token = next_ids[0];

        // Tokenizer 解码
        std::string token_text;
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