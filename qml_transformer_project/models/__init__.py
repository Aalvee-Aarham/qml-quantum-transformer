"""
Model registry for Quantum and Classical Transformer Architectures.
Provides:
  - ModelA: Classical Transformer Baseline (dim-10 bottleneck)
  - ModelB: Hybrid Quantum Transformer (Ideal 10-qubit PennyLane VQC)
  - ModelC: Hybrid Quantum Transformer (Decoherence-simulated 10-qubit VQC, std=0.05)
  - ModelD: Hybrid Quantum Transformer (Physical QPU hardware noise simulation via Qiskit Aer FakeBrisbane)
"""

from models.model_a import ModelA
from models.model_b import ModelB
from models.model_c import ModelC
from models.model_d import ModelD

__all__ = ["ModelA", "ModelB", "ModelC", "ModelD"]
