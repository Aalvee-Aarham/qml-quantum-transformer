"""
Model C: Decoherence-Simulated Hybrid Quantum Transformer.
Architecture:
  - Identical 10-Qubit PennyLane VQC architecture as Model B (depth=2).
  - Simulates environmental decoherence and thermal relaxation by injecting
    zero-mean Gaussian noise (std=0.05) directly into the quantum output expectation values
    during the training phase.
  - Linear logit head (10 -> vocab_size).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from transformer import TransformerBackbone
from models.model_b import create_vqc_circuit, N_QUBITS, CIRCUIT_DEPTH


class ModelC(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 32,
        n_heads: int = 2,
        n_qubits: int = N_QUBITS,
        depth: int = CIRCUIT_DEPTH,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.name = f"Model C (Decoherence std={noise_std} - {n_qubits} Qubits)"
        self.n_qubits = n_qubits
        self.depth = depth
        self.noise_std = noise_std

        self.backbone = TransformerBackbone(vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads)
        self.proj_in = nn.Linear(embed_dim, n_qubits)

        circuit = create_vqc_circuit(n_qubits, depth)
        weight_shapes = {"weights": (depth, 2, n_qubits)}
        self.vqc = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.head = nn.Linear(n_qubits, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Args:
            idx: (B, T) LongTensor
            targets: (B, T) Optional LongTensor
        Returns:
            logits: (B, T, vocab_size)
            loss: cross-entropy loss if targets is provided
        """
        B, T = idx.size()
        h = self.backbone(idx)                    # (B, T, 32)
        
        # Scale to [-\pi, \pi] for angle encoding
        angles = math.pi * torch.tanh(self.proj_in(h))  # (B, T, 10)
        
        angles_flat = angles.view(B * T, self.n_qubits)  # (B*T, 10)
        q_features_flat = self.vqc(angles_flat)          # (B*T, 10)
        q_features = q_features_flat.view(B, T, self.n_qubits)  # (B, T, 10)

        # Inject Gaussian decoherence perturbation during training
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(q_features) * self.noise_std
            # Clamp to physical expectation range [-1, 1]
            q_features = torch.clamp(q_features + noise, -1.0, 1.0)

        logits = self.head(q_features)            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss
