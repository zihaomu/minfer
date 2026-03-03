import numpy as np
import os
from llama_cpp import Llama

ROOT_PATH = "./test/layers/test_data/data"

def export_intermediate():
    model_path = "test/big_models/Lite-Oute-1-65M-FP16.gguf"
    
    # Unfortunately llama_cpp python bindings do not expose intermediate representations easily
    # We will build a small torch or manual comparison if needed. 
    # For now, let's just make sure the token sequence in C++ is right.
    
    # Write a quick script to read the weights and compute just the embedding in numpy
    # to compare against the C++ embedding output.
    pass

if __name__ == "__main__":
    export_intermediate()
