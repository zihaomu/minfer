import numpy as np
import os
from llama_cpp import Llama

ROOT_PATH = "./test/layers/test_data/data"
os.makedirs(ROOT_PATH, exist_ok=True)

def generate_network_data():
    model_path = "test/big_models/Lite-Oute-1-65M-FP16.gguf"
    print(f"Loading model: {model_path}")
    
    # Load model. We need access to logits, so set logits_all=True
    llm = Llama(
        model_path=model_path,
        n_ctx=256,
        logits_all=True,
        verbose=False
    )
    
    # Test prompts
    prompts = [
        "Hello",
        "The quick brown fox",
        "What is the capital of France?",
        "1, 2, 3,"
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i}: '{prompt}'")
        tokens = llm.tokenize(prompt.encode("utf-8"))
        
        # We need to feed these tokens into the model and get the logits for the last token
        # Using eval to just run a forward pass
        llm.eval(tokens)
        
        # In llama.cpp, to get logits we do:
        # Evaluate each token sequence to get full logits matrix
        
        # Re-initialize to clear KV cache for pure calculation
        llm.reset()
        
        # array needs to be [1, seq_len] for minfer input test shape match
        input_ids = np.array(tokens, dtype=np.int32).reshape(1, len(tokens))
        print("Tokens shape:", input_ids.shape)
        
        # Save input
        np.save(f"{ROOT_PATH}/net_input_{i}.npy", input_ids)
        
        # evaluate the tokens
        llm.eval(tokens)
        
        # Get logits sequence
        # _score is expected to be [seq_len, vocab_size]
        logits = []
        # llama-cpp-python eval evaluates the tokens, we can get logits from the state
        # llama_get_logits gets logits for the last token by default if we simply call eval
        
        # A more straightforward way in llama_cpp:
        # Evaluate step by step or all at once, llm._state holds the evaluations
        
        # Llama-cpp-python returns a generator for __call__, but we want raw logits.
        # After eval(tokens), llm.eval outputs logits internally. 
        # We can fetch them via llm._ctx.get_logits() or similar, 
        # but let's use the standard completion API with max_tokens=1 and logprobs to be safe.
        
        # Actually llm.eval() fills the internal logits array. 
        # llm.logits reads the logits for the *last* evaluated token.
        # But if we need logits for all tokens, we need to read the full array if logits_all is True.
        
        # Fetch exactly the sequence of tokens evaluated
        seq_len = len(tokens)
        last_token_logits = np.array(llm.scores[:seq_len]).reshape(1, seq_len, -1)
        
        print("Output logits shape (last token):", last_token_logits.shape)
        np.save(f"{ROOT_PATH}/net_output_{i}.npy", last_token_logits)

if __name__ == "__main__":
    generate_network_data()
