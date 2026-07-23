from tokenizers import Tokenizer

def load_tokenizer(path):
    tokenizer = Tokenizer.from_file(path)
    return tokenizer

def encode(tokenizer, text):
    return tokenizer.encode(text).ids

def decode(tokenizer, ids):
    return tokenizer.decode(ids)