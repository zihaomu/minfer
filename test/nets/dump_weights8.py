import os
import numpy as np
import gguf

model_path = "test/big_models/Lite-Oute-1-65M-FP16.gguf"

reader = gguf.GGUFReader(model_path)
def get_tensor(name):
    for tensor in reader.tensors:
        if tensor.name == name:
            return np.array(tensor.data, dtype=np.float32)
    return None

attn_norm_w = get_tensor("blk.0.attn_norm.weight")
wq_w_raw = get_tensor("blk.0.attn_q.weight")

# Token 0 embedding
tok_0_emb = get_tensor("token_embd.weight")[1 * 512 : 2 * 512] 

# RMSNorm
rms_eps = 1e-06
var = np.mean(tok_0_emb ** 2)
normed_tok_0 = tok_0_emb * (1.0 / np.sqrt(var + rms_eps))
norm_scaled = normed_tok_0 * attn_norm_w

# Compare inner product mechanisms
wq_w = wq_w_raw.reshape(512, 512) 
print(f"Python Reshaped WQ shape: {wq_w.shape}, first 5 elements: {wq_w[0, :5]}")

# GGUF tensor format is usually column-major or row-major depending on transpose. Let's see shapes.
# In C++, WQ first 5: 0.0843506 0.0930786 ...
print(f"Python WQ flat first 5: {wq_w_raw[:5]}")

# The C++ says "transposeB=true", so it computes x_q[i] = dot(x, wq_w[i, :])
x_q_transposed = np.dot(norm_scaled, wq_w.T) 
print(f"Python dot(x, W.T) First 5: {x_q_transposed[:5]}")

x_q_not_transposed = np.dot(norm_scaled, wq_w)
print(f"Python dot(x, W) First 5: {x_q_not_transposed[:5]}")

