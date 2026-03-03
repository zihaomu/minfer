import numpy as np
import gguf

reader = gguf.GGUFReader("test/big_models/Lite-Oute-1-65M-FP16.gguf")
tensors = {t.name: np.array(t.data, dtype=np.float32) for t in reader.tensors}

arr = tensors["token_embd.weight"].reshape((-1, 512))
t0 = arr[1]
t1 = arr[22557]
attn_norm_w = tensors["blk.0.attn_norm.weight"]

t0_n = t0 * (1.0/np.sqrt(np.mean(t0**2) + 1e-6)) * attn_norm_w
t1_n = t1 * (1.0/np.sqrt(np.mean(t1**2) + 1e-6)) * attn_norm_w

wq_w = tensors["blk.0.attn_q.weight"].reshape((512, 512))
wk_w = tensors["blk.0.attn_k.weight"].reshape((256, 512))
wv_w = tensors["blk.0.attn_v.weight"].reshape((256, 512))
wout_w = tensors["blk.0.attn_output.weight"].reshape((512, 512))

xq0 = np.dot(t0_n, wq_w.T)
xq1 = np.dot(t1_n, wq_w.T)
xk0 = np.dot(t0_n, wk_w.T)
xk1 = np.dot(t1_n, wk_w.T)
xv0 = np.dot(t0_n, wv_w.T)
xv1 = np.dot(t1_n, wv_w.T)

xq = np.stack([xq0, xq1]).reshape(2, 16, 32).transpose(1, 0, 2)
xk = np.stack([xk0, xk1]).reshape(2, 8, 32).transpose(1, 0, 2)
xv = np.stack([xv0, xv1]).reshape(2, 8, 32).transpose(1, 0, 2)

qkv_heads = []
for h in range(16):
    h_kv = h // 2
    q = xq[h].copy()
    k = xk[h_kv].copy()
    v = xv[h_kv]
    
    # Dynamic RoPE
    dim = 32
    theta = 10000.0 ** (-2 * np.arange(0, dim//2) / dim)
    freqs_0 = 0.0 * theta
    freqs_1 = 1.0 * theta
    
    sin_0, cos_0 = np.repeat(np.sin(freqs_0), 2), np.repeat(np.cos(freqs_0), 2)
    sin_1, cos_1 = np.repeat(np.sin(freqs_1), 2), np.repeat(np.cos(freqs_1), 2)
    
    # Token 0
    q_r, q_i = q[0, 0::2], q[0, 1::2]
    q[0, 0::2] = q_r * np.cos(freqs_0) - q_i * np.sin(freqs_0)
    q[0, 1::2] = q_r * np.sin(freqs_0) + q_i * np.cos(freqs_0)
    
    k_r, k_i = k[0, 0::2], k[0, 1::2]
    k[0, 0::2] = k_r * np.cos(freqs_0) - k_i * np.sin(freqs_0)
    k[0, 1::2] = k_r * np.sin(freqs_0) + k_i * np.cos(freqs_0)

    # Token 1
    q_r, q_i = q[1, 0::2], q[1, 1::2]
    q[1, 0::2] = q_r * np.cos(freqs_1) - q_i * np.sin(freqs_1)
    q[1, 1::2] = q_r * np.sin(freqs_1) + q_i * np.cos(freqs_1)
    
    k_r, k_i = k[1, 0::2], k[1, 1::2]
    k[1, 0::2] = k_r * np.cos(freqs_1) - k_i * np.sin(freqs_1)
    k[1, 1::2] = k_r * np.sin(freqs_1) + k_i * np.cos(freqs_1)
    
    score = np.matmul(q, k.T) / np.sqrt(32.0)
    if h == 0:
        print("Python score Head 0 before mask:", score[1])
    score[0, 1] = -1e20
    score = np.exp(score - np.max(score, axis=-1, keepdims=True))
    score = score / np.sum(score, axis=-1, keepdims=True)
    
    if h == 0:
        print("Python score Head 0:", score[1])
    
    qkv = np.matmul(score, v)
    qkv_heads.append(qkv)

qkvT = np.stack(qkv_heads).transpose(1, 0, 2).reshape(2, 512)
x_out = np.matmul(qkvT, wout_w.T)
x_out_wrong = np.matmul(qkvT, wout_w)
print("Python x_out Token 1 First 4:", x_out[1, :4])
print("Python x_out W/O TRANSPOSE Token 1 First 4:", x_out_wrong[1, :4])
