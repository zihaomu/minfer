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

attn_norm_w = get_tensor("blk.0.attn_norm.weight").flatten()
wq_w_raw = get_tensor("blk.0.attn_q.weight")
wk_w_raw = get_tensor("blk.0.attn_k.weight")

# Print loaded shapes to debug dimension crash!
print(f"attn_norm_w shape: {attn_norm_w.shape}")
print(f"wq_w_raw shape: {wq_w_raw.shape}")
print(f"wk_w_raw shape: {wk_w_raw.shape}")

# Token 0 embedding is token ID 1 based on earlier tokenization trace
embd = get_tensor("token_embd.weight")
print(f"embd shape: {embd.shape}")
tok_1_emb = embd[22557, :]
print(f"tok_1_emb shape: {tok_1_emb.shape}")

# RMSNorm
rms_eps = 1e-06
var = np.mean(tok_1_emb ** 2)
normed_tok_1 = tok_1_emb * (1.0 / np.sqrt(var + rms_eps))
norm_scaled = normed_tok_1 * attn_norm_w

# Q, K projections
wq_w = wq_w_raw.reshape(512, 512) 
wk_w = wk_w_raw.reshape(256, 512)

x_q_transposed = np.dot(norm_scaled, wq_w.T) 
x_k_transposed = np.dot(norm_scaled, wk_w.T)

n_head = 16
n_head_kv = 8
head_dim = 32
rope_dim = 32
freq_base = 10000.0

xq_heads = x_q_transposed.reshape((n_head, head_dim))
xk_heads = x_k_transposed.reshape((n_head_kv, head_dim))

# Output to verify
print(f"Python reshaped xq_heads[0] First 10: {xq_heads[0, :10]}")

pos = 1 # Token 1
freqs = (1.0 / (freq_base ** (np.arange(0, rope_dim, 2)[: (rope_dim // 2)] / rope_dim)))
freqs_matrix = np.outer(pos, freqs) # (1, rope_dim/2)
freqs_cos = np.cos(freqs_matrix)
freqs_sin = np.sin(freqs_matrix)

# apply RoPE Llama-style
def apply_rope(x_heads, rope_dim):
    x_out = np.zeros_like(x_heads)
    for h in range(x_heads.shape[0]):
        head = x_heads[h]
        for i in range(0, rope_dim, 2):
            x0 = head[i]
            x1 = head[i+1]
            cos_t = freqs_cos[0, i//2]
            sin_t = freqs_sin[0, i//2]
            x_out[h, i]   = x0 * cos_t - x1 * sin_t
            x_out[h, i+1] = x0 * sin_t + x1 * cos_t
        for i in range(rope_dim, head_dim):
            x_out[h, i] = head[i]
    return x_out

xk_rope = apply_rope(xk_heads, rope_dim)
xq_rope = apply_rope(xq_heads, rope_dim)

print("Python RoPE xq_heads[0] First 10:", np.round(xq_rope[0, :10], 8))
print("Python RoPE xk_heads[0] First 10:", np.round(xk_rope[0, :10], 8))

qk_head = np.dot(xq_heads[0].flatten(), xk_heads[0].flatten())
print("Python QK Head 0:", qk_head / np.sqrt(32.0))
