"""
Model B: Hybrid Quantum Transformer (Ideal PennyLane VQC Baseline).
Architecture:
  - TransformerBackbone (embed_dim=32, n_heads=2)
  - Linear feature projection to 10 rotation angles
  - 10-Qubit PennyLane Variational Quantum Circuit (depth=2):
      * RY state encoding
      * Strongly entangling variational layers (RY + RZ + CNOT cyclic ladder)
      * Pauli-Z expectation measurements on all 10 qubits
  - Linear logit head (10 -> vocab_size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from transformer import TransformerBackbone

N_QUBITS = 10
CIRCUIT_DEPTH = 2

# Initialize PennyLane statevector device
dev_ideal = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev_ideal, interface="torch", diff_method="backprop")
def ideal_vqc_circuit(inputs, weights):
    """
    10-Qubit VQC with RY state encoding, depth=2 RY/RZ ansatz, and CNOT entanglement.
    Args:
        inputs: (10,) tensor of input rotation angles
        weights: (depth, 2, 10) tensor of variational rotation parameters
    Returns:
        List of Pauli-Z expectation values across all 10 qubits in range [-1, 1].
    """
    # 1. RY State Encoding (using [..., i] for tensor batch broadcasting)
    for i in range(N_QUBITS):
        qml.RY(inputs[..., i], wires=i)

    # 2. Variational Ansatz Layers (depth = 2)
    for d in range(CIRCUIT_DEPTH):
        for i in range(N_QUBITS):
            qml.RY(weights[d, 0, i], wires=i)
            qml.RZ(weights[d, 1, i], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CNOT(wires=[i, i + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])  # Cyclic boundary entanglement

    # 3. Pauli-Z expectation measurements
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def create_vqc_circuit(n_qubits: int = N_QUBITS, depth: int = CIRCUIT_DEPTH):
    """Dynamically creates an ideal PennyLane VQC QNode for any given number of qubits."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def vqc(inputs, weights):
        # State encoding
        for i in range(n_qubits):
            qml.RY(inputs[..., i], wires=i)
        # Variational ansatz
        for d in range(depth):
            for i in range(n_qubits):
                qml.RY(weights[d, 0, i], wires=i)
                qml.RZ(weights[d, 1, i], wires=i)
            if n_qubits > 1:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return vqc

# Backward compatibility default 10-qubit circuit
ideal_vqc_circuit = create_vqc_circuit(N_QUBITS, CIRCUIT_DEPTH)


class ModelB(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32, n_heads: int = 2, n_qubits: int = N_QUBITS, depth: int = CIRCUIT_DEPTH):
        super().__init__()
        self.name = f"Model B (Hybrid Quantum Transformer - {n_qubits} Qubits)"
        self.n_qubits = n_qubits
        self.depth = depth

        self.backbone = TransformerBackbone(vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads)
        self.proj_in = nn.Linear(embed_dim, n_qubits)

        # Dynamic VQC circuit for any qubit count
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
        
        # Scale intermediate features into [-\pi, \pi] for angle encoding
        angles = math.pi * torch.tanh(self.proj_in(h))  # (B, T, 10)
        
        # Flatten sequence into batch dimension for efficient vectorized circuit execution
        angles_flat = angles.view(B * T, self.n_qubits)  # (B*T, 10)
        q_features_flat = self.vqc(angles_flat)          # (B*T, 10)
        q_features = q_features_flat.view(B, T, self.n_qubits)  # (B, T, 10)

        logits = self.head(q_features)            # (B, T, vocab_size)

        loss = None
        if targets is not None:
            V = logits.size(-1)
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss
