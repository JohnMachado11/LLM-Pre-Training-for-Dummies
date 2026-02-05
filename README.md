# LLM-Pre-Training-for-Dummies

This book teaches you how to pre-train a GPT-2 language model from scratch on a single GPU. No abstractions, no hand-waving. You'll understand every line of code and every design decision.

By the end, you'll know:

- How tokenization converts text to numbers using Byte Pair Encoding
- How embeddings turn token IDs into learnable vector representations
- How self-attention lets tokens communicate with each other
- Why we scale dot products and apply causal masking
- How multi-head attention runs parallel attention operations
- How transformer blocks combine attention, feed-forward networks, and residual connections
- How the training loop works: forward pass, cross-entropy loss, backpropagation, AdamW
- How to make training efficient with mixed precision, gradient accumulation, and Flash Attention
- How to prepare the OpenWebText dataset and run a real training job
- How to monitor loss curves, save checkpoints, and generate text

This is the book version of Andrej Karpathy's nanoGPT. Same philosophy: minimal code, maximum understanding. Everything runs on a single RTX 3090 or similar 24GB GPU.

No fine-tuning. No RLHF. Just pre-training, done right.

## Who This Is For

You have basic PyTorch knowledge (tensors). You want to understand how language models actually work at the implementation level. You're not afraid of matrix multiplications.

## Links

- Book: https://elliotarledge.gumroad.com/l/pretraining-for-dummies
- Reference repo: https://github.com/Infatoshi/pre-training-for-dummies