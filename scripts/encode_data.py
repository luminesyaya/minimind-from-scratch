import os
import numpy as np
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/pretrain_clean/tokenizer.json")
def encode_file(input_path, output_path):
    all_ids = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                ids = tokenizer.encode(line).ids
                all_ids.extend(ids)
    arr = np.array(all_ids, dtype=np.uint16)
    arr.tofile(output_path)
    return len(arr)

train_tokens = encode_file("data/pretrain_clean/train.txt",
                           "data/pretrain_clean/train.bin")
valid_tokens = encode_file("data/pretrain_clean/valid.txt",
                           "data/pretrain_clean/valid.bin")

print(f"Train tokens: {train_tokens}")
print(f"Valid tokens: {valid_tokens}")