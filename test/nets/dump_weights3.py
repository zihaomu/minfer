import gguf
import numpy as np

reader = gguf.GGUFReader("test/big_models/Lite-Oute-1-65M-FP16.gguf")
norm_weights = None
for tensor in reader.tensors:
    if tensor.name == 'blk.0.attn_norm.weight':
        norm_weights = tensor.data.view(np.float32)
        break

emb_full = None
for tensor in reader.tensors:
    if tensor.name == 'token_embd.weight':
        tok_weights = tensor.data.view(np.float16).astype(np.float32).reshape((32768, 512))
        emb_full = tok_weights[1] # Token 0 is ID 1
        break

# Compare exact loop math in Python vs C++
sum_f2 = 0.0
for j in range(512):
    sum_f2 += emb_full[j] * emb_full[j]

rms_eps = 1e-5
x1_cpp_style = 1.0 / np.sqrt(sum_f2 / 512.0 + rms_eps)

out_cpp_style = np.zeros(512, dtype=np.float32)
for j in range(512):
    out_cpp_style[j] = emb_full[j] * x1_cpp_style * norm_weights[j]

print("Python-emulated C++ RMSNorm Token 1 First 10:", out_cpp_style[:10])
print("sum_f2/512 =", sum_f2/512)

# Now, C++ printed:
# EmbLayer Token 0 (ID 1) First 10: 0.18457 -0.0446777 -0.00642776 -0.0396729 -0.0770264 0.0562439 -0.0678101 -0.0412903 0.0551453 -0.0348206
print("\nPython Emb Token 1 First 10:", emb_full[:10])

