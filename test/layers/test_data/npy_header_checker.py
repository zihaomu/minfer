import numpy as np
import numpy.lib.format as fmt

file_path = "./data/word_emb_params_0.npy"

with open(file_path, "rb") as f:
    magic = f.read(6)
    version = f.read(2)
    header_len = int.from_bytes(f.read(2), 'little')  # 1.0 header长度，2字节
    header_bytes = f.read(header_len)
    print(header_bytes.decode('latin1'))
