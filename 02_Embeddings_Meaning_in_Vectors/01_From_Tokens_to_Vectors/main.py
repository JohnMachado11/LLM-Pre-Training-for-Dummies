import torch
import torch.nn as nn


vocab_size = 50257
n_embd = 768
# This creates a table of shape (50257, 768) - about 38 million parameters.
# Each row is a learnable vector representing one token.
embedding = nn.Embedding(vocab_size, n_embd)

print(embedding) # Embedding(50257, 768)

# Given a batch of token IDs with shape (batch_size, sequence_length), the embedding layer returns
# vectors of shape (batch_size, sequence_length, n_embd).
# The input has shape (1, 4) - one sequence of four tokens. 
# The output has shape (1, 4, 768) - four 768-dimensional vectors.
idx = torch.tensor([[15496, 11, 995, 0]])
print(idx.shape) # torch.Size([1, 4])

emb = embedding(idx)
print(emb)
print(emb.shape) # torch.Size([1, 4, 768])