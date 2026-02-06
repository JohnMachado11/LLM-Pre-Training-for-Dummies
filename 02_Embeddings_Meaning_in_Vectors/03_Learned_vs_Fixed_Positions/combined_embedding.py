import torch
import torch.nn as nn


class CombinedEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(0.1)
    
    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        return self.drop(tok + pos)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_size = 1024
n_emb = 768
vocab_size = 50257
n_embd = 768

embed = CombinedEmbedding(vocab_size, block_size, n_embd).to(device)

# Example batch of 2 sequences, each of length 4 (token IDs)
idx = torch.tensor([
    [15496, 11, 995, 0],   # "Hello, world!" in GPT-2 ids (common demo)
    [464, 2068, 318, 257], # another random-ish GPT-2-ish sequence
], dtype=torch.long, device=device)

x = embed(idx)  # (B, T, C)
print("idx.shape =", idx.shape)  # (2, 4)
print("x.shape   =", x.shape)    # (2, 4, 768)
print("first token combined first 5 =", x[0, 0, :5])
