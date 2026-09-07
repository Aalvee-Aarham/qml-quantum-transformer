"""
Model A: Classical Baseline Transformer.
Architecture:
  - TransformerBackbone (embed_dim=32, n_heads=2)
  - Linear Bottleneck layer (dim 10, matching the 10-qubit quantum feature scale)
  - Linear Logit Head (10 -> vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TransformerBackbone


class ModelA(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, n_heads: int = 2, bottleneck_dim: int = 10):
        super().__init__()
        self.name = f"Model A (Classical Baseline - Dim {bottleneck_dim})"
        self.backbone = TransformerBackbone(vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads)
        self.bottleneck = nn.Linear(embed_dim, bottleneck_dim)
        self.activation = nn.Tanh()  # Bounds features to [-1, 1], matching Pauli-Z expectation range
        self.head = nn.Linear(bottleneck_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: (B, T) LongTensor of token indices
            targets: (B, T) Optional LongTensor of target next-tokens
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar cross-entropy loss if targets is provided, else None
        """
        h = self.backbone(idx)              # (B, T, 32)
        features = self.activation(self.bottleneck(h))  # (B, T, 10)
        logits = self.head(features)        # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.size()
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss
