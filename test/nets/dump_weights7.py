import sys
import os
import numpy as np

# Load ground truth from python reference implementation output generated for net_tiny_llama test
print("Loading npy...")
try:
    py_logits = np.load('test/layers/test_data/data/net_output_0.npy')
    print(f"PyTorch reference out_logits shape: {py_logits.shape}")
    print(f"PyTorch first 10 logits token 0: {py_logits[0, 0, :10]}")
    if py_logits.shape[1] > 1:
        print(f"PyTorch first 10 logits token 1: {py_logits[0, 1, :10]}")
except Exception as e:
    print(e)
