import numpy as np
import os
from llama_cpp import Llama

ROOT_PATH = "./test/layers/test_data/data"
os.makedirs(ROOT_PATH, exist_ok=True)

def debug_network_data():
    model_path = "test/big_models/Lite-Oute-1-65M-FP16.gguf"
    print(f"Loading model: {model_path}")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=256,
        logits_all=True,
        verbose=False
    )
    
    prompt = "Hello"
    print(f"\nProcessing prompt: '{prompt}'")
    tokens = llm.tokenize(prompt.encode("utf-8"))
    
    llm.reset()
    input_ids = np.array(tokens, dtype=np.int32)
    np.save(f"{ROOT_PATH}/debug_input.npy", input_ids)
    print("Tokens shape:", input_ids.shape)
    
    # Evaluate
    llm.eval(tokens)
    
    # We can get embeddings and other outputs from the state if we wanted
    # but initially let's just make sure the token sequence matches, and the final state is dumped
    last_token_logits = np.array(llm.scores[:])
    np.save(f"{ROOT_PATH}/debug_output.npy", last_token_logits)
    print("Scores shape:", last_token_logits.shape)
    print("Sample scores:", last_token_logits[0, :5])

if __name__ == "__main__":
    debug_network_data()
