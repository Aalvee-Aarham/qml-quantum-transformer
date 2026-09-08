"""
Model E - Real IBM Quantum Hardware Evaluation
================================================

Evaluates your ALREADY-TRAINED Model D checkpoint by running its 10-qubit VQC on a
real IBM QPU instead of AerSimulator(FakeBrisbane). Results are stored under the
"Model E" label so they can be merged directly into the experiment CSVs alongside
Models A-D without overwriting existing data.

Place this file in your project root (next to main.py, dataset.py, transformer.py,
trainer.py, models/).

AUTH
----
This script reads your IBM Quantum API key from the .env file (project root) or
from the environment variable IBM_QUANTUM_TOKEN. It is never hardcoded, so it is
safe to commit this file / share it without leaking your credentials.

    # Fill in .env at the project root:
    #   IBM_QUANTUM_TOKEN=your_api_key_here
    #   IBM_QUANTUM_INSTANCE=your_instance_crn_here   (optional)

    # Then run (Windows PowerShell / any shell):
    python model_e_runner.py

    # Or set env vars directly (Windows PowerShell):
    $env:IBM_QUANTUM_TOKEN = "your_api_key_here"
    python model_e_runner.py

Get your API key from https://quantum.cloud.ibm.com (Dashboard -> API keys).

WHAT THIS DOES AND DOES NOT DO
-------------------------------
- Loads the classical weights (backbone, proj_in, head) and quantum weights
  (analytic_vqc.weights) from your saved checkpoints/Model_D_*.pt file.
- Replays your training and validation batches through that FROZEN model, but with
  the VQC executed on real hardware instead of a simulator.
- "Train_Loss" below means "loss of training-split batches evaluated through the
  real QPU", NOT a new gradient-descent training run.
- Before any QPU time is spent, the script prints exactly how many circuits/jobs
  it's about to submit and asks you to confirm.
- After a successful hardware run, results are automatically merged into
  thesis_data_collection.csv and qubit_scaling_data.csv (10-qubit row only).
"""

import os
import sys
import json
import time
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load .env file from project root before anything else
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if not _env_path.exists():
        _env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_env_path)
    print(f"[Env] Loaded .env from: {_env_path}")
except ImportError:
    print("[Env] python-dotenv not installed. Falling back to system environment variables.")
    print("      Install with: pip install python-dotenv")

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #

PROJECT_DIR = Path(__file__).resolve().parent   # qml_transformer_project/
sys.path.insert(0, str(PROJECT_DIR))

from dataset import ShakespeareDataModule
from transformer import TransformerBackbone

N_QUBITS    = 10
DEPTH       = 2
EMBED_DIM   = 32
N_HEADS     = 2
SHOTS       = 128

SEQ_LEN     = 16
BATCH_SIZE  = 16
VAL_SPLIT   = 0.1
DATA_SEED   = 42

# Start small to verify the script works before spending real QPU minutes.
NUM_TRAIN_BATCHES = 5
NUM_VAL_BATCHES   = 10

# None = auto-detect the newest Model D checkpoint under checkpoints/
CHECKPOINT_PATH = None

# Raw physical measurement (no automatic error mitigation).
# Set to 1 or 2 if you'd rather report mitigated numbers.
RESILIENCE_LEVEL = 0

OUTPUT_DIR = PROJECT_DIR / "ibm_model_e_results"

SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# 2. IBM Quantum authentication (key comes from .env / environment)
# --------------------------------------------------------------------------- #

def get_ibm_service() -> QiskitRuntimeService:
    token    = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")  # optional CRN

    if token:
        print("[Auth] Using IBM_QUANTUM_TOKEN from environment / .env file.")
        kwargs = {"channel": "ibm_quantum_platform", "token": token}
        if instance:
            kwargs["instance"] = instance
            print(f"[Auth] Using IBM_QUANTUM_INSTANCE: {instance}")
        return QiskitRuntimeService(**kwargs)

    print("[Auth] IBM_QUANTUM_TOKEN not set -- falling back to a previously saved account.")
    print("       (Run QiskitRuntimeService.save_account(...) once, separately, if this fails.)")
    return QiskitRuntimeService()


def select_backend(service: QiskitRuntimeService, min_qubits: int = N_QUBITS):
    backend = service.least_busy(min_num_qubits=min_qubits, operational=True, simulator=False)
    print("Selected backend:      ", backend.name)
    print("Physical qubits:       ", backend.num_qubits)
    print("Operational:           ", backend.status().operational)
    try:
        print("Pending jobs in queue: ", backend.status().pending_jobs)
    except Exception:
        pass
    if backend.num_qubits < min_qubits:
        raise RuntimeError(
            f"Backend has only {backend.num_qubits} qubits; Model E needs {min_qubits}."
        )
    return backend


# --------------------------------------------------------------------------- #
# 3. Checkpoint loading (+ simulator fallback if none found)
# --------------------------------------------------------------------------- #

def find_model_d_checkpoint(project_dir: Path):
    """Find the most recently modified Model D (or Model E) checkpoint."""
    ckpt_dir = project_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    candidates = [
        p for p in ckpt_dir.glob("*.pt")
        if any(tok in p.name for tok in ("Model_D", "Model_E", "Qiskit", "FakeBrisbane"))
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def create_vqc_circuit_pennylane(n_qubits, depth):
    """PennyLane analytic circuit -- used only for the simulator fallback."""
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for d in range(depth):
            for i in range(n_qubits):
                qml.RY(weights[d, 0, i], wires=i)
                qml.RZ(weights[d, 1, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[n_qubits - 1, 0])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class ModelESim(nn.Module):
    """
    Simulator fallback: trains a fresh Model-D-style VQC on PennyLane so the rest
    of the script has something real to evaluate on hardware when no checkpoint exists.
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, n_heads=N_HEADS,
                 n_qubits=N_QUBITS, depth=DEPTH):
        super().__init__()
        import pennylane as qml

        self.backbone   = TransformerBackbone(vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads)
        self.proj_in    = nn.Linear(embed_dim, n_qubits)
        circuit         = create_vqc_circuit_pennylane(n_qubits, depth)
        self.analytic_vqc = qml.qnn.TorchLayer(circuit, {"weights": (depth, 2, n_qubits)})
        self.head       = nn.Linear(n_qubits, vocab_size)
        self.n_qubits   = n_qubits

    def forward(self, idx, targets=None):
        B, T = idx.size()
        h      = self.backbone(idx)
        angles = math.pi * torch.tanh(self.proj_in(h))
        q      = self.analytic_vqc(angles.view(B * T, self.n_qubits)).view(B, T, self.n_qubits)
        logits = self.head(q)
        loss   = None
        if targets is not None:
            V    = logits.size(-1)
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss


def load_or_train_checkpoint(data_module):
    """Returns (state_dict, checkpoint_path_or_None, meta_dict)."""
    checkpoint_path = CHECKPOINT_PATH or find_model_d_checkpoint(PROJECT_DIR)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        print("Using checkpoint:", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        meta = {
            "model_name":       checkpoint.get("model_name", "unknown"),
            "stored_val_loss":  checkpoint.get("val_loss"),
            "stored_perplexity": checkpoint.get("perplexity"),
        }
        print("Checkpoint model:  ", meta["model_name"])
        print("Stored val loss:   ", meta["stored_val_loss"])
        print("Stored perplexity: ", meta["stored_perplexity"])
        return state_dict, checkpoint_path, meta

    print("No Model D/E checkpoint found under checkpoints/ -- training a fresh one on the")
    print("simulator now (fast, free). Run main.py first if you want to evaluate your")
    print("actual trained checkpoint instead.")

    sim_model = ModelESim(vocab_size=data_module.vocab_size)
    optimizer  = torch.optim.AdamW(sim_model.parameters(), lr=0.003, weight_decay=1e-2)
    for step in range(1, 151):
        sim_model.train()
        x, y = data_module.get_batch(split="train")
        optimizer.zero_grad()
        _, loss = sim_model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sim_model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % 30 == 0 or step == 150:
            print(f"  [sim-train] step {step}/150 loss={loss.item():.4f}")

    meta = {"model_name": "Model E (simulator fallback, no checkpoint found)"}
    print("Fallback simulator training complete.")
    return sim_model.state_dict(), None, meta


# --------------------------------------------------------------------------- #
# 4. Classical shell reconstruction
# --------------------------------------------------------------------------- #

class ModelEClassicalShell(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_qubits=N_QUBITS):
        super().__init__()
        self.backbone = TransformerBackbone(vocab_size=vocab_size, embed_dim=embed_dim, n_heads=n_heads)
        self.proj_in  = nn.Linear(embed_dim, n_qubits)
        self.head     = nn.Linear(n_qubits, vocab_size)


def build_classical_shell_and_qweights(state_dict, vocab_size):
    model_shell = ModelEClassicalShell(vocab_size=vocab_size)

    classical_state = {
        k: v for k, v in state_dict.items()
        if k.startswith("backbone.") or k.startswith("proj_in.") or k.startswith("head.")
    }
    missing, unexpected = model_shell.load_state_dict(classical_state, strict=False)
    if missing:
        raise RuntimeError(f"Missing classical checkpoint keys: {missing}")
    model_shell.eval()
    print("Classical weights loaded. Unexpected keys ignored:", unexpected)

    q_weight_key = "analytic_vqc.weights"
    if q_weight_key not in state_dict:
        raise KeyError(f"Could not find {q_weight_key!r} in checkpoint state_dict.")

    q_weights = state_dict[q_weight_key].detach().cpu().numpy().astype(np.float64)
    expected_shape = (DEPTH, 2, N_QUBITS)
    if tuple(q_weights.shape) != expected_shape:
        raise ValueError(f"Quantum weight shape {q_weights.shape} != expected {expected_shape}.")
    print("Quantum weight shape:", q_weights.shape)

    # Only requires_grad parameters count -- buffers (e.g. causal_mask) must be excluded.
    classical_trainable = sum(p.numel() for p in model_shell.parameters() if p.requires_grad)
    trainable_params    = classical_trainable + q_weights.size
    print("Trainable parameters (classical + quantum, buffers excluded):", trainable_params)

    return model_shell, q_weights, trainable_params


# --------------------------------------------------------------------------- #
# 5. VQC circuit + transpilation
# --------------------------------------------------------------------------- #

def build_model_e_circuit(n_qubits=N_QUBITS, depth=DEPTH):
    """Builds the parameterised Qiskit circuit matching the PennyLane VQC topology."""
    x  = ParameterVector("x", n_qubits)
    w  = ParameterVector("w", depth * 2 * n_qubits)
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.ry(x[i], i)
    for d in range(depth):
        offset = d * (2 * n_qubits)
        for i in range(n_qubits):
            qc.ry(w[offset + i], i)
            qc.rz(w[offset + n_qubits + i], i)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.cx(n_qubits - 1, 0)
    qc.measure_all()
    return qc


def transpile_for_backend(backend):
    template_circuit = build_model_e_circuit()
    pm               = generate_preset_pass_manager(backend=backend, optimization_level=1)
    start            = time.perf_counter()
    isa_circuit      = pm.run(template_circuit)
    transpile_seconds = time.perf_counter() - start
    print("Transpilation time:", round(transpile_seconds, 3), "s")
    print("ISA circuit depth:  ", isa_circuit.depth())
    print("ISA circuit ops:    ", isa_circuit.count_ops())
    print("ISA parameters:     ", len(isa_circuit.parameters))
    return isa_circuit, transpile_seconds


# --------------------------------------------------------------------------- #
# 6. Real-hardware executor (one parameter-sweep PUB per batch)
# --------------------------------------------------------------------------- #

def make_parameter_matrix(angles_np, weights_np, isa_circuit):
    """Build the (N, num_isa_params) array in the exact order the transpiler expects."""
    N      = angles_np.shape[0]
    values = {f"x[{i}]": angles_np[:, i] for i in range(N_QUBITS)}
    flat_w = weights_np.reshape(-1)
    for j, v in enumerate(flat_w):
        values[f"w[{j}]"] = np.full(N, v, dtype=np.float64)

    rows = np.zeros((N, len(isa_circuit.parameters)), dtype=np.float64)
    for col, param in enumerate(isa_circuit.parameters):
        name = param.name
        if name in values:
            rows[:, col] = values[name]
        elif name.startswith("x") and name[1:].isdigit():
            rows[:, col] = angles_np[:, int(name[1:])]
        elif name.startswith("w") and name[1:].isdigit():
            rows[:, col] = flat_w[int(name[1:])]
        else:
            raise KeyError(f"Could not map ISA parameter: {name!r}")
    return rows


def counts_to_z_expectations(counts, n_qubits=N_QUBITS):
    shots   = sum(counts.values())
    expvals = np.zeros(n_qubits, dtype=np.float64)
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(n_qubits):
            bit = int(bits[n_qubits - 1 - q])
            expvals[q] += (1.0 if bit == 0 else -1.0) * count
    expvals /= float(shots)
    return expvals.astype(np.float32)


def make_sampler(backend):
    sampler = Sampler(mode=backend)
    try:
        sampler.options.resilience_level = RESILIENCE_LEVEL
    except Exception as e:
        print(f"Could not set resilience_level via options ({e}); using backend default.")
    print("Sampler ready. resilience_level =", RESILIENCE_LEVEL)
    return sampler


def run_hardware_batch(sampler, isa_circuit, angles_np, weights_np, shots=SHOTS):
    parameter_matrix = make_parameter_matrix(angles_np, weights_np, isa_circuit)
    start            = time.perf_counter()
    job              = sampler.run([(isa_circuit, parameter_matrix)], shots=shots)
    print("  submitted job:", job.job_id())
    result           = job.result()
    elapsed          = time.perf_counter() - start
    bit_array        = result[0].data.meas
    q_features       = np.zeros((angles_np.shape[0], N_QUBITS), dtype=np.float32)
    for k in range(angles_np.shape[0]):
        q_features[k] = counts_to_z_expectations(bit_array.get_counts(k))
    return q_features, job.job_id(), elapsed


# --------------------------------------------------------------------------- #
# 7. Forward pass + evaluation loop with spend-confirmation gate
# --------------------------------------------------------------------------- #

@torch.no_grad()
def get_qpu_angles(model_shell, idx):
    h      = model_shell.backbone(idx)
    angles = math.pi * torch.tanh(model_shell.proj_in(h))
    return angles.reshape(-1, N_QUBITS).cpu().numpy().astype(np.float64)


@torch.no_grad()
def hardware_forward_and_loss(model_shell, sampler, isa_circuit, q_weights, x, y):
    angles_np  = get_qpu_angles(model_shell, x)
    q_features, job_id, elapsed = run_hardware_batch(
        sampler, isa_circuit, angles_np, q_weights, shots=SHOTS
    )
    q_features_t = torch.tensor(q_features, dtype=torch.float32).view(
        x.size(0), x.size(1), N_QUBITS
    )
    logits = model_shell.head(q_features_t)
    V      = logits.size(-1)
    loss   = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
    return loss.item(), job_id, elapsed


def evaluate_split(data_module, model_shell, sampler, isa_circuit,
                   q_weights, split, num_batches):
    losses, jobs, wall_times = [], [], []
    print(f"\n===== REAL QPU {split.upper()} EVALUATION (Model E) =====")
    for b in range(num_batches):
        x, y = data_module.get_batch(split=split)
        loss, job_id, elapsed = hardware_forward_and_loss(
            model_shell, sampler, isa_circuit, q_weights, x, y
        )
        losses.append(loss)
        jobs.append(job_id)
        wall_times.append(elapsed)
        print(
            f"[{split}] batch {b + 1:02d}/{num_batches:02d} | "
            f"loss={loss:.6f} | job={job_id} | time={elapsed:.2f}s"
        )
    return {
        "mean_loss":       float(np.mean(losses)),
        "std_loss":        float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
        "losses":          losses,
        "jobs":            jobs,
        "wall_times":      wall_times,
        "total_wall_time": float(np.sum(wall_times)),
    }


# --------------------------------------------------------------------------- #
# 8. Result export + CSV / graph integration
# --------------------------------------------------------------------------- #

def save_results(backend, train_result, val_result, total_runtime,
                 trainable_params, checkpoint_path, transpile_seconds):
    train_loss_hw = train_result["mean_loss"]
    val_loss_hw   = val_result["mean_loss"]
    perplexity_hw = math.exp(val_loss_hw)

    print("\n===== FINAL REAL-QPU METRICS (Model E) =====")
    print("Train loss (hardware-eval):", train_loss_hw)
    print("Validation loss:           ", val_loss_hw)
    print("Perplexity:                ", perplexity_hw)
    print("Total wall-clock runtime:  ", total_runtime, "s")

    result_row = {
        "Qubits":                N_QUBITS,
        "Hilbert_Dim":           2 ** N_QUBITS,
        "Type":                  "Real QPU",
        "Model":                 f"Model E (Real IBM Quantum QPU - {N_QUBITS} Qubits)",
        "Train_Loss":            round(train_loss_hw, 4),
        "Val_Loss":              round(val_loss_hw, 4),
        "Perplexity":            round(perplexity_hw, 4),
        "Trainable_Params":      trainable_params,
        "Runtime_Seconds":       round(total_runtime, 2),
        "Backend":               backend.name,
        "Backend_Physical_Qubits": backend.num_qubits,
        "Shots":                 SHOTS,
        "Resilience_Level":      RESILIENCE_LEVEL,
        "Train_Batches":         NUM_TRAIN_BATCHES,
        "Val_Batches":           NUM_VAL_BATCHES,
        "Checkpoint":            str(checkpoint_path) if checkpoint_path else "simulator fallback",
        "Transpile_Seconds":     round(transpile_seconds, 3),
        "QPU_Jobs":              len(train_result["jobs"]) + len(val_result["jobs"]),
        "Evaluation_Mode":       (
            "Frozen trained Model D evaluated on real QPU (not gradient-trained on hardware)"
        ),
    }
    df_result = pd.DataFrame([result_row])
    print(df_result.T)

    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path  = OUTPUT_DIR / "model_e_real_qpu_statistics.csv"
    json_path = OUTPUT_DIR / "model_e_real_qpu_provenance.json"
    df_result.to_csv(csv_path, index=False)

    provenance = {
        "backend":                       backend.name,
        "physical_qubits":               backend.num_qubits,
        "logical_qubits":                N_QUBITS,
        "hilbert_dimension":             2 ** N_QUBITS,
        "depth":                         DEPTH,
        "shots":                         SHOTS,
        "resilience_level":              RESILIENCE_LEVEL,
        "train_losses_per_batch":        train_result["losses"],
        "validation_losses_per_batch":   val_result["losses"],
        "train_job_ids":                 train_result["jobs"],
        "validation_job_ids":            val_result["jobs"],
        "train_batch_wall_times":        train_result["wall_times"],
        "validation_batch_wall_times":   val_result["wall_times"],
        "total_runtime_seconds":         total_runtime,
        "transpile_seconds":             transpile_seconds,
        "checkpoint":                    str(checkpoint_path) if checkpoint_path else None,
        "model_name":                    result_row["Model"],
        "note": (
            "Train_Loss/Val_Loss are hardware-evaluated losses of a fixed, previously trained "
            "Model D (used as Model E). This script does not perform end-to-end gradient "
            "training on real hardware."
        ),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print("CSV saved:        ", csv_path)
    print("Provenance saved: ", json_path)

    merge_into_thesis_csv(df_result)
    merge_into_scaling_csv(df_result)
    regenerate_thesis_curves()
    regenerate_scaling_plot()

    return df_result


# --------------------------------------------------------------------------- #
# 9. Merge Model E results into existing experiment CSVs
# --------------------------------------------------------------------------- #

def merge_into_thesis_csv(df_result):
    """
    Appends the Model E 10-qubit row to thesis_data_collection.csv,
    replacing any existing 'Model E' row so re-runs don't duplicate data.
    """
    thesis_columns = ["Model", "Train_Loss", "Val_Loss", "Perplexity",
                       "Trainable_Params", "Runtime_Seconds"]
    candidate_paths = [
        PROJECT_DIR / "thesis_data_collection.csv",
        PROJECT_DIR.parent / "thesis_data_collection.csv",
    ]
    existing_csv = next((p for p in candidate_paths if p.exists()), None)

    if existing_csv is None:
        print("[Merge] No thesis_data_collection.csv found -- skipping thesis merge.")
        return

    old = pd.read_csv(existing_csv)
    # Remove any pre-existing Model E row to avoid duplicates on re-run
    old = old[~old["Model"].str.contains("Model E", na=False)]

    available_cols = [c for c in thesis_columns if c in df_result.columns]
    new_row = df_result[available_cols].copy()
    updated = pd.concat([old, new_row], ignore_index=True)

    # Save back to all copies
    for path in candidate_paths:
        if path.parent.exists():
            updated.to_csv(path, index=False)
            print(f"[Merge] thesis_data_collection.csv updated: {path}")
    print(updated[["Model", "Train_Loss", "Val_Loss", "Perplexity"]].to_string(index=False))


def merge_into_scaling_csv(df_result):
    """
    Appends the Model E 10-qubit row to qubit_scaling_data.csv under Type='Real QPU',
    replacing any existing 'Real QPU' row at N=10 so re-runs don't duplicate.
    """
    scaling_columns = ["Qubits", "Hilbert_Dim", "Type", "Model", "Train_Loss",
                        "Val_Loss", "Perplexity", "Trainable_Params", "Runtime_Seconds"]
    candidate_paths = [
        PROJECT_DIR / "qubit_scaling_data.csv",
        PROJECT_DIR.parent / "qubit_scaling_data.csv",
    ]
    existing_csv = next((p for p in candidate_paths if p.exists()), None)

    if existing_csv is None:
        print("[Merge] No qubit_scaling_data.csv found -- skipping scaling merge.")
        return

    old = pd.read_csv(existing_csv)
    # Remove any pre-existing Real QPU rows at N=10 to avoid duplicates
    old = old[~((old["Type"] == "Real QPU") & (old["Qubits"] == N_QUBITS))]

    available_cols = [c for c in scaling_columns if c in df_result.columns]
    new_row = df_result[available_cols].copy()
    updated = pd.concat([old, new_row], ignore_index=True)

    for path in candidate_paths:
        if path.parent.exists():
            updated.to_csv(path, index=False)
            print(f"[Merge] qubit_scaling_data.csv updated: {path}")
    print(updated[["Qubits", "Type", "Val_Loss", "Perplexity"]].to_string(index=False))


# --------------------------------------------------------------------------- #
# 10. Graph regeneration
# --------------------------------------------------------------------------- #

def _get_short_name(full_name: str, colors: dict) -> str:
    for k in colors:
        if k in full_name:
            return k
    return full_name[:7]


def regenerate_thesis_curves():
    """
    Regenerates thesis_loss_curves.png to include a Model E bar in the
    Perplexity and Runtime panels (no step-level loss history for hardware eval).
    """
    candidate_paths = [
        PROJECT_DIR / "thesis_data_collection.csv",
        PROJECT_DIR.parent / "thesis_data_collection.csv",
    ]
    existing_csv = next((p for p in candidate_paths if p.exists()), None)
    if existing_csv is None:
        print("[Graph] thesis_data_collection.csv not found -- skipping thesis curve regeneration.")
        return

    df = pd.read_csv(existing_csv)
    if df.empty:
        return

    colors = {
        "Model A": "#1f77b4",
        "Model B": "#2ca02c",
        "Model C": "#ff7f0e",
        "Model D": "#9467bd",
        "Model E": "#d62728",  # distinct red for real QPU
    }

    def short_name(full):
        return _get_short_name(full, colors)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.patch.set_facecolor("#fafbfc")
    fig.suptitle(
        "Experimental Summary: Validation Perplexity & Runtime (incl. Model E Real IBM QPU)",
        fontsize=13, fontweight="bold", y=1.02
    )

    names       = [short_name(m) for m in df["Model"]]
    ppls        = df["Perplexity"].tolist()
    runtimes    = df["Runtime_Seconds"].tolist()
    params      = df["Trainable_Params"].tolist()
    bar_colors  = [colors.get(n, "#555555") for n in names]

    # Panel 1: Perplexity
    ax1   = axes[0]
    bars1 = ax1.bar(names, ppls, color=bar_colors, edgecolor="#222222", width=0.55, alpha=0.88)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_title("Validation Perplexity (exp(Val Loss))", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 2: Runtime
    ax2   = axes[1]
    bars2 = ax2.bar(names, runtimes, color=bar_colors, edgecolor="#222222", width=0.55, alpha=0.85)
    for idx, bar in enumerate(bars2):
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}s\n({params[idx]:,}p)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_title("Total Runtime & Parameter Efficiency", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Runtime (Seconds)", fontsize=11)
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(pad=3.0)
    for path in [PROJECT_DIR / "thesis_loss_curves.png",
                 PROJECT_DIR.parent / "thesis_loss_curves.png"]:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[Graph] thesis_loss_curves.png regenerated: {path}")
    plt.close()


def regenerate_scaling_plot():
    """
    Regenerates qubit_scaling_analysis.png to include the Real QPU (Model E)
    line at N=10 alongside the existing 4-model sweep.
    """
    candidate_paths = [
        PROJECT_DIR / "qubit_scaling_data.csv",
        PROJECT_DIR.parent / "qubit_scaling_data.csv",
    ]
    existing_csv = next((p for p in candidate_paths if p.exists()), None)
    if existing_csv is None:
        print("[Graph] qubit_scaling_data.csv not found -- skipping scaling plot regeneration.")
        return

    df = pd.read_csv(existing_csv)
    if df.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#fafbfc")

    colors = {
        "Classical":   "#1f77b4",
        "Ideal Quantum": "#2ca02c",
        "Decoherence": "#ff7f0e",
        "Physical QPU": "#9467bd",
        "Real QPU":    "#d62728",   # Model E
    }
    markers = {
        "Classical":    "s",
        "Ideal Quantum": "o",
        "Decoherence":  "^",
        "Physical QPU": "D",
        "Real QPU":     "*",
    }
    styles = {
        "Classical":    "--",
        "Ideal Quantum": "-",
        "Decoherence":  "-.",
        "Physical QPU": ":",
        "Real QPU":     (0, (3, 1, 1, 1)),  # dense dash-dot
    }

    model_types = ["Classical", "Ideal Quantum", "Physical QPU", "Decoherence", "Real QPU"]
    qubits_all  = sorted(df["Qubits"].unique())

    # Panel 1: Perplexity vs Qubits
    ax1 = axes[0, 0]
    for mtype in model_types:
        sub = df[df["Type"] == mtype].sort_values("Qubits")
        if sub.empty:
            continue
        ax1.plot(sub["Qubits"], sub["Perplexity"],
                 marker=markers.get(mtype, "o"),
                 linestyle=styles.get(mtype, "-"),
                 linewidth=2.2, markersize=8 if mtype == "Real QPU" else 7,
                 color=colors.get(mtype, "#333"), label=mtype)
        for _, row in sub.iterrows():
            offset = 6 if mtype in ["Ideal Quantum", "Physical QPU", "Real QPU"] else -12
            ax1.annotate(f"{row['Perplexity']:.1f}",
                         (row["Qubits"], row["Perplexity"]),
                         textcoords="offset points", xytext=(0, offset),
                         ha="center", fontsize=8, fontweight="bold",
                         color=colors.get(mtype, "#333"))
    ax1.set_title("Validation Perplexity vs. Qubit Count (All Models)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax1.set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    ax1.set_xticks([2, 4, 6, 8, 10])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Loss vs Qubits
    ax2 = axes[0, 1]
    for mtype in model_types:
        sub = df[df["Type"] == mtype].sort_values("Qubits")
        if sub.empty:
            continue
        ax2.plot(sub["Qubits"], sub["Val_Loss"],
                 marker=markers.get(mtype, "o"),
                 linestyle=styles.get(mtype, "-"),
                 linewidth=2.2, markersize=8 if mtype == "Real QPU" else 7,
                 color=colors.get(mtype, "#333"), label=mtype)
    ax2.set_title("Validation Loss (Cross-Entropy) vs. Qubit Count", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.set_xticks([2, 4, 6, 8, 10])
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Runtime Scaling (Log Scale)
    ax3    = axes[1, 0]
    x_idx  = np.arange(len(qubits_all))
    width  = 0.15
    n_types = len(model_types)
    for i, mtype in enumerate(model_types):
        sub = df[df["Type"] == mtype].sort_values("Qubits")
        if sub.empty:
            continue
        runtimes = [sub[sub["Qubits"] == q]["Runtime_Seconds"].values[0]
                    if not sub[sub["Qubits"] == q].empty else 0
                    for q in qubits_all]
        offsets  = (i - (n_types - 1) / 2) * width
        ax3.bar(x_idx + offsets, runtimes, width=width,
                color=colors.get(mtype, "#333"), alpha=0.85,
                edgecolor="#222", label=mtype)
    ax3.set_xticks(x_idx)
    ax3.set_xticklabels([f"N={q}" for q in qubits_all])
    ax3.set_yscale("log")
    ax3.set_title("Training / Evaluation Runtime Comparison (Log Scale)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax3.set_ylabel("Runtime in Seconds (Log Scale)", fontsize=11)
    ax3.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 4: Noise Penalty (Δ Perplexity over Ideal VQC)
    ax4     = axes[1, 1]
    q_ideal = df[df["Type"] == "Ideal Quantum"].set_index("Qubits")["Perplexity"]
    q_decoh = df[df["Type"] == "Decoherence"].set_index("Qubits")["Perplexity"]
    q_qpu   = df[df["Type"] == "Physical QPU"].set_index("Qubits")["Perplexity"]
    q_real  = df[df["Type"] == "Real QPU"].set_index("Qubits")["Perplexity"]

    common_q = [q for q in qubits_all
                if q in q_ideal.index and q in q_decoh.index and q in q_qpu.index]
    if common_q:
        decoh_gap = [q_decoh.loc[q] - q_ideal.loc[q] for q in common_q]
        qpu_gap   = [q_qpu.loc[q]   - q_ideal.loc[q] for q in common_q]
        bar_w     = 0.25
        ax4.bar(np.array(common_q) - bar_w, decoh_gap, width=bar_w,
                color=colors["Decoherence"], alpha=0.85, edgecolor="#222",
                label="Gaussian Decoherence Penalty (+Δ PPL)")
        ax4.bar(np.array(common_q),          qpu_gap,   width=bar_w,
                color=colors["Physical QPU"], alpha=0.85, edgecolor="#222",
                label="IBM FakeBrisbane QPU Noise Penalty (+Δ PPL)")

    # Overlay Real QPU point at N=10 if available
    real_common = [q for q in [10] if q in q_ideal.index and q in q_real.index]
    if real_common:
        real_gap = [q_real.loc[q] - q_ideal.loc[q] for q in real_common]
        bar_w    = 0.25
        ax4.bar(np.array(real_common) + bar_w, real_gap, width=bar_w,
                color=colors["Real QPU"], alpha=0.85, edgecolor="#222",
                label="Real IBM QPU Noise Penalty (+Δ PPL)")

    ax4.set_title("Noise Penalty over Ideal VQC (+Δ Perplexity)", fontsize=12, fontweight="bold", pad=10)
    ax4.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax4.set_ylabel("Perplexity Increase (Lower is Better)", fontsize=11)
    ax4.set_xticks(common_q if common_q else [10])
    ax4.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(pad=3.0)
    for path in [PROJECT_DIR / "qubit_scaling_analysis.png",
                 PROJECT_DIR.parent / "qubit_scaling_analysis.png"]:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[Graph] qubit_scaling_analysis.png regenerated: {path}")
    plt.close()


# --------------------------------------------------------------------------- #
# 11. Main
# --------------------------------------------------------------------------- #

def main():
    set_seed(SEED)

    service = get_ibm_service()
    print("Available instances:", service.instances())
    backend = select_backend(service, min_qubits=N_QUBITS)

    data_module = ShakespeareDataModule(
        seq_len=SEQ_LEN, batch_size=BATCH_SIZE, val_split=VAL_SPLIT, seed=DATA_SEED
    )
    vocab_size = data_module.vocab_size
    print("Vocabulary size:", vocab_size)

    state_dict, checkpoint_path, _meta = load_or_train_checkpoint(data_module)
    model_shell, q_weights, trainable_params = build_classical_shell_and_qweights(
        state_dict, vocab_size
    )

    isa_circuit, transpile_seconds = transpile_for_backend(backend)
    sampler = make_sampler(backend)

    total_jobs = NUM_TRAIN_BATCHES + NUM_VAL_BATCHES
    print(
        f"\nAbout to submit {total_jobs} jobs total "
        f"({NUM_TRAIN_BATCHES} train + {NUM_VAL_BATCHES} val batches, "
        f"{BATCH_SIZE * SEQ_LEN} circuits/job, {SHOTS} shots each) "
        f"to real backend '{backend.name}'."
    )
    print("This consumes part of your IBM Quantum runtime-minute quota.")
    confirm = input("Type 'yes' to submit to real hardware, anything else to abort: ")

    if confirm.strip().lower() != "yes":
        print("Aborted -- no jobs were sent to IBM hardware.")
        return

    experiment_start = time.perf_counter()
    train_result = evaluate_split(
        data_module, model_shell, sampler, isa_circuit, q_weights, "train", NUM_TRAIN_BATCHES
    )
    val_result = evaluate_split(
        data_module, model_shell, sampler, isa_circuit, q_weights, "val", NUM_VAL_BATCHES
    )
    total_runtime = time.perf_counter() - experiment_start

    save_results(backend, train_result, val_result, total_runtime,
                 trainable_params, checkpoint_path, transpile_seconds)

    print("\n--- How to describe this in your experimental report --------------------------------")
    print("DO say: 'Model E evaluates the frozen Model D VQC weights on a real IBM")
    print("Quantum processor. The resulting hardware measurements are passed through")
    print("the frozen classical output head to compute hardware-evaluated cross-entropy")
    print("loss and perplexity.'")
    print(f"Resilience_Level={RESILIENCE_LEVEL} means no automatic error mitigation was")
    print("applied -- this is a raw physical measurement.")


if __name__ == "__main__":
    main()
