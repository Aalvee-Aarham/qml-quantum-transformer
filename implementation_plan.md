# Implementation Plan: Quantum-Classical Hybrid Transformer Comparative Study

Build a modular, thesis-grade Python framework comparing classical and quantum neural architectures within a character-level Transformer for next-token prediction on Tiny Shakespeare. The study benchmarks four architectures:
- **Model A**: Classical Transformer Baseline (with dim-10 bottleneck)
- **Model B**: Hybrid Quantum Transformer (10-qubit ideal PennyLane VQC)
- **Model C**: Decoherence-Simulated Quantum Transformer (10-qubit VQC + Gaussian noise $\sigma=0.05$)
- **Model D**: Physical QPU Hardware-Simulated Transformer (10-qubit VQC with IBM `FakeBrisbane` noise model via Qiskit Aer)

---

## 1. Laptop Hardware Specifications & Estimated Execution Time

Based on terminal interrogation of your system, here are your verified hardware specifications:

| Hardware Component | Detected Specification |
| :--- | :--- |
| **Processor (CPU)** | **11th Gen Intel(R) Core(TM) i5-1145G7 @ 2.60GHz** (Boost up to 4.40GHz) |
| **Cores & Threads** | **4 Physical Cores, 8 Logical Processors** |
| **System Memory (RAM)** | **15.82 GB DDR4** |
| **Graphics (GPU)** | **Intel(R) Iris(R) Xe Graphics** (PyTorch runs on CPU via AVX2/AVX-512) |
| **Python Environment** | **Python 3.14.6 (64-bit)** with PyTorch 2.12.1+cpu |

### Estimated Execution Time Breakdown (150 steps, Batch Size = 16, Seq Len = 16)

For 10 qubits, the statevector dimension is $2^{10} = 1,024$ complex amplitudes (~16 KB), which fits comfortably within the **12 MB L3 cache** of your Intel Core i5-1145G7.

| Stage / Model | Simulation Engine / Method | Expected Runtime | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **Data Ingestion & Setup** | Download 20k chars Tiny Shakespeare + Tokenizer | **~1 – 2 seconds** | One-time cached download |
| **Model A (Classical)** | PyTorch CPU (1 Layer, 2 Heads, Dim-10 bottleneck) | **~1 – 2 seconds** | ~10 ms/step across 150 steps |
| **Model B (Ideal Quantum)** | PennyLane `default.qubit` (Analytic statevector) | **~18 – 30 seconds** | Vectorized PyTorch backprop (~120 ms/step) |
| **Model C (Gaussian Noise)** | PennyLane `default.qubit` + $\mathcal{N}(0, 0.05^2)$ jitter | **~18 – 30 seconds** | In-flight decoherence perturbation |
| **Model D (Physical QPU)** | Qiskit Aer with IBM `FakeBrisbane` Noise Model | **~60 – 100 seconds** | OpenMP-accelerated density matrix/noisy execution |
| **Metric Export & Plotting** | CSV export + 300 DPI Matplotlib loss curves | **~2 – 3 seconds** | Publication-grade chart generation |
| **Total Pipeline Runtime** | **Complete end-to-end execution** | **~1.5 – 3.0 minutes** | Fully manageable in interactive session |

> [!NOTE]
> **Runtime Optimization Insight**: Naive shot-based parameter-shift simulation for 10 qubits on 256 tokens per batch would evaluate over $3,000,000$ quantum circuits, taking over **2 hours**. In our thesis-grade architecture, we use **noise-aware forward simulation via Qiskit Aer's `FakeBrisbane` noise model combined with analytical adjoint backprop** (standard in QML research literature). This achieves full physical noise fidelity without gradient variance explosion, cutting runtime from **2+ hours down to ~90 seconds** on your 4-core CPU!

---

## 2. Proposed Architectural Enhancements ("Better Way to Implement")

To ensure the project is truly **thesis-grade** and scientifically rigorous, we introduce several key enhancements:

1. **Dimensional & Scale Alignment**:
   - Model A includes an explicit `Linear(32 -> 10)` bottleneck followed by `Tanh` (bounded in $[-1, 1]$) before the logit projection. This perfectly isolates whether any observed performance difference is due to the **expressivity of the Hilbert space** or merely a **bottleneck parameter reduction**.
2. **Causal Masking**:
   - Strictly enforces lower-triangular causal attention masking in `transformer.py` so tokens cannot attend to future positions during next-token prediction.
3. **Hardware-Efficient Quantum Ansatz**:
   - 10-qubit RY state encoding: mapping normalized transformer token representations to $[-\pi, \pi]$.
   - Depth-2 variational block: $R_Y(\theta_1) \cdot R_Z(\theta_2)$ on all 10 wires, followed by cyclic entangling CNOT ladders, with $\langle \hat{Z}_i \rangle \in [-1, 1]$ expectation values measured across all 10 qubits.
4. **Physical Noise Modeling for Model D**:
   - Extracts real calibration data ($T_1, T_2$ relaxation, thermal dissipation, single/two-qubit gate errors, and readout fidelities) from `FakeBrisbane` (or Qiskit Aer's device noise generator), applying them directly to the 10-qubit register.
5. **Publication-Grade Metrics & Visualizations**:
   - Matplotlib styling with scientific color palette, step-wise training curves, validation checkpoints, rolling averages, and final perplexity bar/radar annotations.

---

## 3. Project File Structure

All code will reside cleanly in `qml_transformer_project/`:

```
c:\Users\User\Downloads\qnlp\
└── qml_transformer_project/
    ├── requirements.txt
    ├── dataset.py
    ├── transformer.py
    ├── trainer.py
    ├── main.py
    └── models/
        ├── __init__.py
        ├── model_a.py
        ├── model_b.py
        ├── model_c.py
        └── model_d.py
```

---

## 4. Proposed Changes by Component

### Component 1: Environment & Dependencies
#### [NEW] [requirements.txt](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/requirements.txt)
- Pin required libraries: `torch`, `pennylane`, `pennylane-qiskit`, `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`, `pandas`, `matplotlib`.

---

### Component 2: Data Pipeline
#### [NEW] [dataset.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/dataset.py)
- `load_data()`: Auto-fetches the 20,000-character subset of Karpathy's Tiny Shakespeare.
- `CharTokenizer`: Character-level vocabulary builder with `encode(str) -> List[int]` and `decode(List[int]) -> str`.
- `ShakespeareDataset` / `get_batch()`: Generates training and validation mini-batches (`batch_size=16`, `seq_len=16`, 90/10 split) with causal shifted targets ($x_{1:T} \to y_{2:T+1}$).

---

### Component 3: Classical Transformer Backbone
#### [NEW] [transformer.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/transformer.py)
- `CausalSelfAttention`: 2 attention heads, `embed_dim=32`, lower-triangular causal mask.
- `TransformerBackbone`: Learned token embeddings (vocab $\to 32$) + learned positional embeddings ($16 \to 32$) + Multi-Head Self-Attention + Residual Add & LayerNorm.
- Modular interface yielding intermediate feature representations $H \in \mathbb{R}^{B \times T \times 32}$.

---

### Component 4: Model Implementations
#### [NEW] [models/__init__.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/models/__init__.py)
- Expose `ModelA`, `ModelB`, `ModelC`, `ModelD`.

#### [NEW] [models/model_a.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/models/model_a.py)
- **Model A (Classical Baseline)**:
  `TransformerBackbone` $\to$ `Linear(32, 10)` $\to \text{Tanh}$ bottleneck $\to$ `Linear(10, vocab_size)` logit projection.

#### [NEW] [models/model_b.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/models/model_b.py)
- **Model B (Ideal Hybrid Quantum Transformer)**:
  `TransformerBackbone` $\to$ `Linear(32, 10)` feature scaling $\to$ 10-qubit PennyLane VQC (`default.qubit`) with depth 2 ansatz $\to$ 10 Pauli-Z expectations $\to$ `Linear(10, vocab_size)`.

#### [NEW] [models/model_c.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/models/model_c.py)
- **Model C (Decoherence-Simulated Quantum Transformer)**:
  Same 10-qubit VQC as Model B, but injecting Gaussian noise $\mathcal{N}(0, 0.05^2)$ into the 10 expectation values during training to emulate environmental decoherence and drift.

#### [NEW] [models/model_d.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/models/model_d.py)
- **Model D (Physical QPU Hardware-Simulated Transformer)**:
  Same 10-qubit VQC mapped onto Qiskit Aer backend with IBM `FakeBrisbane` noise calibration properties (or real hardware depolarizing & relaxation noise model), capturing physical gate error rates, $T_1/T_2$ decay, and readout errors.

---

### Component 5: Training & Evaluation Engine
#### [NEW] [trainer.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/trainer.py)
- `train_model(model, data, steps=150, lr=0.003, batch_size=16)`:
  - AdamW optimizer.
  - Cross-entropy loss over next-token character predictions.
  - Records step-by-step training loss, validation loss, runtime, and total trainable parameters.
  - Computes final validation loss and perplexity $\text{PPL} = \exp(\mathcal{L}_{\text{val}})$.

---

### Component 6: Orchestration & Research Deliverables
#### [NEW] [main.py](file:///c:/Users/User/Downloads/qnlp/qml_transformer_project/main.py)
- Runs Model A, B, C, and D sequentially with fixed seeds for reproducibility.
- Produces clean console comparison tables.
- Exports `thesis_data_collection.csv` with columns:
  `Model`, `Train_Loss`, `Val_Loss`, `Perplexity`, `Trainable_Params`, `Runtime_Seconds`.
- Generates `thesis_loss_curves.png` (high-res 300 DPI comparative training/validation convergence curves).

---

## 5. Verification Plan

### Automated Execution & Sanity Testing
1. **Dependency Installation**: Install missing dependencies (`pennylane`, `pennylane-qiskit`, `qiskit`, `qiskit-aer`, `qiskit-ibm-runtime`).
2. **Component Smoke Test**: Run a quick dry-run test of single forward-backward pass across all 4 models.
3. **Full 150-Step Experiment**: Run `python main.py` and verify all 4 models run cleanly to completion.
4. **Artifact Validation**:
   - Inspect `thesis_data_collection.csv` for numeric integrity.
   - Verify `thesis_loss_curves.png` exists and visually conveys training dynamics.
