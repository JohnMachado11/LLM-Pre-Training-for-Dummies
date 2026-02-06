import torch
import torch.nn as nn


class CombinedEmbedding(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, debug=False, max_print=5):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(0.1)
        self.debug = debug
        self.max_print = max_print

    def forward(self, idx):

        # idx has shape (B, T):
        #   B = batch size (number of sequences)
        #   T = sequence length (tokens per sequence)
        # After embedding lookup, tok has shape (B, T, C):
        #   C = embedding dimension (n_embd)
        B, T = idx.shape

        if self.debug:
            print(f"\n[CombinedEmbedding] idx.shape={tuple(idx.shape)} dtype={idx.dtype} device={idx.device}")
            show_t = min(T, self.max_print)
            print(f"[CombinedEmbedding] idx[0, :{show_t}] =", idx[0, :show_t].detach().cpu().tolist())

        tok = self.tok_emb(idx)  # (B, T, C)

        pos_ids = torch.arange(T, device=idx.device)  # (T,)
        pos = self.pos_emb(pos_ids)  # (T, C)

        x = tok + pos  # broadcasts (T,C) over batch -> (B,T,C)
        x_drop = self.drop(x)

        if self.debug:
            print(f"[CombinedEmbedding] tok.shape={tuple(tok.shape)} pos.shape={tuple(pos.shape)}")
            print(f"[CombinedEmbedding] x.shape={tuple(x.shape)} x_drop.shape={tuple(x_drop.shape)}")
            print("[CombinedEmbedding] tok[0,0,:5] =", tok[0, 0, :5].detach().cpu().tolist())
            print("[CombinedEmbedding] pos[0,:5]   =", pos[0, :5].detach().cpu().tolist())
            print("[CombinedEmbedding] x[0,0,:5]   =", x[0, 0, :5].detach().cpu().tolist())
            print("[CombinedEmbedding] x_drop[0,0,:5] =", x_drop[0, 0, :5].detach().cpu().tolist())

        return x_drop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_size = 1024
n_emb = 768
vocab_size = 50257
n_embd = 768

embed = CombinedEmbedding(vocab_size, block_size, n_embd, debug=True).to(device)

# Example batch of 2 sequences, each of length 4 (token IDs)
idx = torch.tensor([
    [15496, 11, 995, 0],   # "Hello, world!" in GPT-2 ids (common demo)
    [464, 2068, 318, 257], # another random-ish GPT-2-ish sequence
], dtype=torch.long, device=device)

x = embed(idx)  # (B, T, C)
print("idx.shape =", idx.shape)  # (2, 4)
print("x.shape   =", x.shape)    # (2, 4, 768)
print("first token combined first 5 =", x[0, 0, :5])