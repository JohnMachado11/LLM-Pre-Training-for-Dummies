import torch
import torch.nn as nn


block_size = 1024
n_emb = 768
# This creates a table of shape (1024, 768). Row 0 represents position 0. 
# Row 1 represents position 1. And so on.
pos_embedding = nn.Embedding(block_size, n_emb)
print(pos_embedding, "\n") # Embedding(1024, 768)

vocab_size = 50257
n_embd = 768
# This creates a table of shape (50257, 768) - about 38 million parameters.
# Each row is a learnable vector representing one token.
token_embedding = nn.Embedding(vocab_size, n_embd)

idx = torch.tensor([[15496, 11, 995, 0]])

device = idx.device
seq_len = idx.size(1)

print("Device:", device) # cpu
print("Sequence Length:", seq_len) # 4

# During the forward pass, we add token embeddings and positional embeddings.
pos = torch.arange(0, seq_len, device=device)
print("Positions:", pos, "\n") 

tok_emb = token_embedding(idx)
pos_emb = pos_embedding(pos)
x = tok_emb + pos_emb

# Now the vector at each position carries information about both which token it is and where it
# appears in the sequence.

print("tok_emb first token, first 5 =\n", tok_emb[0, 0, :5], "\n")  # token embedding
print("pos_emb position 0, first 5 =\n", pos_emb[0, :5], "\n")      # position embedding
print("x (tok+pos) first token, first 5 =\n", x[0, 0, :5], "\n")    # combined

print("x.shape =", x.shape)