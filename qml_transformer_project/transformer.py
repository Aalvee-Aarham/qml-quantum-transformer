"""
Transformer backbone architecture for character-level language modeling.
Includes token embedding, positional embedding, causal multi-head self-attention,
and LayerNorm normalization blocks.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention layer with embed_dim=32 and n_heads=2.
    """
    def __init__(self, embed_dim: int = 32, n_heads: int = 2, max_seq_len: int = 64):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        # Causal mask buffer to ensure autoregressive causality
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Compute query, key, values for all heads
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)

        return self.c_proj(y)


class TransformerBackbone(nn.Module):
    """
    1-Layer Transformer Backbone with positional embeddings, LayerNorm,
    and causal self-attention.
    Outputs intermediate hidden states of shape (B, T, embed_dim).
    """
    def __init__(self, vocab_size: int, embed_dim: int = 32, n_heads: int = 2, max_seq_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim=embed_dim, n_heads=n_heads, max_seq_len=max_seq_len)
        self.ln_2 = nn.LayerNorm(embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of shape (B, T)
        Returns:
            Hidden states tensor of shape (B, T, embed_dim)
        """
        B, T = idx.size()
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        positions = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)

        # Token + positional embeddings
        x = self.tok_emb(idx) + self.pos_emb(positions)  # (B, T, embed_dim)

        # Residual attention block
        x = x + self.attn(self.ln_1(x))
        x = self.ln_2(x)

        return x
