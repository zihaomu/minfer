from llama_cpp import Llama

# 1. 加载 GGUF 模型
llm = Llama(
    model_path="/home/moo/work/my_lab/minfer/test/big_models/Lite-Oute-1-65M-FP16.gguf",
    vocab_only=True,
)

# 2. tokenizer 示例
text = "Hi!"
tokens = llm.tokenize(text.encode("utf-8"))
print("Text:", text)
print("Tokens:", tokens)

# 3. detokenizer 示例
decoded = llm.detokenize(tokens).decode("utf-8", errors="ignore")
print("Decoded:", decoded)

# 4. chat 模式
text = "Hello world!"
chat_tokens = llm.apply_chat_template(
    [
        {"role": "user", "content": "Hello world!"},
    ],
    tokenize=True,
    add_generation_prompt=True,
)
print("Chat Tokens:", chat_tokens)

# 5. chat 模式解码
chat_decoded = llm.detokenize(chat_tokens).decode("utf-8", errors="ignore")
print("Chat Decoded:", chat_decoded)
