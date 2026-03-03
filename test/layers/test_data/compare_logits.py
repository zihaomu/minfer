import numpy as np
import os
from llama_cpp import Llama

ROOT_PATH = "./test/layers/test_data/data"

def extract_intermediate():
    model_path = "test/big_models/Lite-Oute-1-65M-FP16.gguf"
    
    llm = Llama(
        model_path=model_path,
        n_ctx=256,
        logits_all=True,
        verbose=False
    )
    
    prompt = "Hello"
    tokens = llm.tokenize(prompt.encode("utf-8"))
    
    llm.reset()
    llm.eval(tokens)
    
    # Just print the exact tokens used
    print("Tokens:", tokens)
    
    # Fetch exactly the first 10 logits of the first token prediction (sequence index 0)
    scores = np.array(llm.scores[:])
    print("Scores shape:", scores.shape)
    
    print("Prompt 0, Token 0 logits (first 10):", scores[0, :10])
    print("Prompt 0, Token 1 logits (first 10):", scores[1, :10])
    
    # Save the proper expected shape [seq_len, vocab_size] sequence targets
    # wait net_output_0 is shape (256, 32768) which implies it dumped the entire 256 ctx instead of just seq_len
    print("True sequence length evaluated:", len(tokens))

if __name__ == "__main__":
    extract_intermediate()
