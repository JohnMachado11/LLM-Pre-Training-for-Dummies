import numpy as np
import tiktoken


enc = tiktoken.get_encoding("gpt2")

print("GPT-2 Vocab Size:", enc.n_vocab) # 50257

text = "Hello, world!"
tokens = enc.encode(text)
print("Tokens:", tokens)

decoded_tokens = enc.decode(tokens)
print("Decoded tokens:", decoded_tokens, "\n")

text = "The quick brown fox jumps over the lazy dog."
char_tokens = list(text)
bpe_tokens = enc.encode(text)

print(f"Characters: {len(char_tokens)} tokens") # 44 tokens
print(f"BPE: {len(bpe_tokens)} tokens") # 10 tokens


# For training, we typically encode the entire dataset once upfront and save it as a binary file of
# token IDs. This avoids re-tokenizing during training.
# We use encode_ordinary instead of encode to skip special token handling. And uint16 is suﬀicient
# since the vocabulary is under 65,536 tokens.
tokens = enc.encode_ordinary(text)

print("\nTokens:", tokens)
decoded_tokens = enc.decode(tokens)
print("Decoded tokens:", decoded_tokens, "\n")

arr = np.array(tokens, dtype=np.uint16)
arr.tofile("train.bin")

# During training, we memory-map this file and sample random chunks.
# Memory mapping lets us access a file larger than RAM without loading it all at once. The OS
# handles paging data in and out as needed.
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
print("Memory Map:", data)