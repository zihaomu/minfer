from llama_cpp import Llama

# 1. 加载 GGUF 模型
llm = Llama(
    model_path=r"E:\\my_project\\minfer\\test\\big_models\\Lite-Oute-1-65M-FP16.gguf",  # 改成你本地模型路径
    vocab_only=True,   # 只加载 tokenizer 和 vocab，不加载权重
)

# 2. tokenizer 示例
text = "Hello world! <s>"
tokens = llm.tokenize(text.encode("utf-8"))
print("Text:", text)
print("Tokens:", tokens)

# 3. detokenizer 示例
decoded = llm.detokenize(tokens).decode("utf-8", errors="ignore")
print("Decoded:", decoded)
