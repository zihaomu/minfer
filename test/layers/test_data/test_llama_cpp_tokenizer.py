from llama_cpp import Llama

# 1. 加载 GGUF 模型
llm = Llama(
    model_path="/home/moo/work/my_lab/minfer/test/big_models/Lite-Oute-1-65M-FP16.gguf"
)

# 2. tokenizer 示例
text = "Who are"  # 这里的文本可以是任何你想测试的内容

# tokens = llm.tokenize(text.encode("utf-8"))
# print("Text:", text)
# print("Tokens:", tokens)

output = llm(text, max_tokens=50)

print(output['choices'][0]['text'])

# tokens = llm.tokenize(text.encode("utf-8"))
# print("Text:", text)
# print("Tokens:", tokens)

# # 3. detokenizer 示例
# decoded = llm.detokenize(tokens).decode("utf-8", errors="ignore")
# print("Decoded:", decoded)

# # 4. chat 模式
# text = "Hi!"
# chat_tokens = llm.create_chat_completion(
#     [
#         {"role": "user", "content": "Hello world! <s>"},
#     ],
#     tokenize=True,
#     add_generation_prompt=True,
# )
# print("Chat Tokens:", chat_tokens)

# # 5. chat 模式解码
# chat_decoded = llm.detokenize(chat_tokens).decode("utf-8", errors="ignore")
# print("Chat Decoded:", chat_decoded)
