import numpy as np
from llama_cpp import Llama

llm = Llama(model_path="test/big_models/Lite-Oute-1-65M-FP16.gguf", n_ctx=256, logits_all=True, verbose=False)
tokens = llm.tokenize(b"Hello")
print("len tokens:", len(tokens))

llm.reset()
llm.eval(tokens)

for i in range(5):
    print(f"row {i}:", llm.scores[i][:5])

# Does llama-cpp-python populate llm.scores up to n_tokens evaluated?
# In version 0.2, if logits_all=True, the entire context has logits.
# BUT are they shifted? The logit for token i is predicting token i+1?
