import gguf
import numpy as np
from llama_cpp import Llama

llm = Llama(model_path="test/big_models/Lite-Oute-1-65M-FP16.gguf", n_ctx=256, verbose=False)
tokens = llm.tokenize(b"Hello")
llm.reset()
llm.eval([tokens[0]]) 
# logit mean should match.
print("Mean logit tok 0:", np.mean(llm.scores[0]))

reader = gguf.GGUFReader("test/big_models/Lite-Oute-1-65M-FP16.gguf")
norm_weights = None
for tensor in reader.tensors:
    if tensor.name == 'blk.0.attn_norm.weight':
        norm_weights = tensor.data.view(np.float32)
        break

# The prompt 0 first token is 1. We captured RMSNorm First 10 inside C++:
# C++: 0.0201976 -0.00509417 -0.00165723 -0.00426197 -0.0121121 0.0188727 -0.012934 -0.00747078 0.0147018 -0.000285175 
# Let's print Python's computed norm for token 1
emb_full = None
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        tok_weights = tensor.data.view(np.float16).astype(np.float32).reshape((32768, 512))
        emb_full = tok_weights[1] # Token 0 is ID 1
        break

variance = np.mean(emb_full ** 2)
inv_std = 1.0 / np.sqrt(variance + 1e-5)
x_norm_true = emb_full * inv_std * norm_weights
print("Python computed RMSNorm Token 1 First 10:", x_norm_true[:10])

for tensor in reader.tensors:
    if tensor.name == 'blk.0.attn_q.weight':
        wq = tensor.data.view(np.float16).astype(np.float32).reshape((512, 512))
        break
x_q_true = np.dot(x_norm_true, wq.T)
print("Python computed x_q Token 1 First 10:", x_q_true[:10])

for tensor in reader.tensors:
    if tensor.name == 'blk.0.attn_k.weight':
        wk = tensor.data.view(np.float16).astype(np.float32).reshape((256, 512))
        break
x_k_true = np.dot(x_norm_true, wk.T)
print("Python computed x_k Token 1 First 10:", x_k_true[:10])

