"""
Comprehensive 4-Model Qubit Scaling Study (N = 2, 4, 6, 8, 10 Qubits).
Benchmarks:
  - Model A: Classical Baseline (Bottleneck Dim = N)
  - Model B: Hybrid Quantum Transformer (Ideal PennyLane VQC, N Qubits)
  - Model C: Decoherence-Simulated Quantum Transformer (N Qubits, std=0.05)
  - Model D: Physical QPU Hardware-Simulated Transformer (IBM FakeBrisbane via Qiskit Aer, N Qubits)
Tracks perplexity, cross-entropy validation loss, parameter count, and CPU execution runtime.
"""

import os
import sys
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Ensure the project package directory is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from dataset import ShakespeareDataModule
from models.model_a import ModelA
from models.model_b import ModelB
from models.model_c import ModelC
from models.model_d import ModelD
from trainer import train_model


def set_seed(seed: int = 42):
    """Sets random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_scaling_analysis(df: pd.DataFrame, save_path: str):
    """
    Generates a 4-panel publication-grade comparison across all 4 models over 2, 4, 6, 8, 10 qubits.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    fig.patch.set_facecolor("#fafbfc")

    colors = {
        "Classical": "#1f77b4",     # Blue
        "Ideal Quantum": "#2ca02c", # Green
        "Decoherence": "#ff7f0e",   # Orange
        "Physical QPU": "#9467bd",  # Purple
        "Real QPU": "#d62728",      # Red — Model E (real IBM hardware)
    }
    markers = {
        "Classical": "s",
        "Ideal Quantum": "o",
        "Decoherence": "^",
        "Physical QPU": "D",
        "Real QPU": "*",
    }
    styles = {
        "Classical": "--",
        "Ideal Quantum": "-",
        "Decoherence": "-.",
        "Physical QPU": ":",
        "Real QPU": (0, (3, 1, 1, 1)),  # dense dash-dot
    }

    model_types = ["Classical", "Ideal Quantum", "Physical QPU", "Decoherence", "Real QPU"]

    # Panel 1: Perplexity vs Qubits
    ax1 = axes[0, 0]
    for mtype in model_types:
        sub = df[df["Type"] == mtype].sort_values("Qubits")
        if not sub.empty:
            ax1.plot(sub["Qubits"], sub["Perplexity"],
                     marker=markers.get(mtype, "o"),
                     linestyle=styles.get(mtype, "-"),
                     linewidth=2.2, markersize=7,
                     color=colors.get(mtype, "#333"), label=mtype)
            for _, row in sub.iterrows():
                offset = 6 if mtype in ["Ideal Quantum", "Physical QPU"] else -12
                ax1.annotate(f"{row['Perplexity']:.1f}", (row["Qubits"], row["Perplexity"]),
                             textcoords="offset points", xytext=(0, offset),
                             ha="center", fontsize=8, fontweight="bold", color=colors.get(mtype, "#333"))

    ax1.set_title("Validation Perplexity vs. Qubit Count (All 4 Models)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax1.set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    ax1.set_xticks([2, 4, 6, 8, 10])
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Loss vs Qubits
    ax2 = axes[0, 1]
    for mtype in model_types:
        sub = df[df["Type"] == mtype].sort_values("Qubits")
        if not sub.empty:
            ax2.plot(sub["Qubits"], sub["Val_Loss"],
                     marker=markers.get(mtype, "o"),
                     linestyle=styles.get(mtype, "-"),
                     linewidth=2.2, markersize=7,
                     color=colors.get(mtype, "#333"), label=mtype)

    ax2.set_title("Validation Loss (Cross-Entropy) vs. Qubit Count", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.set_xticks([2, 4, 6, 8, 10])
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Runtime Scaling vs Qubits (Log Scale)
    ax3 = axes[1, 0]
    qubits = sorted(df["Qubits"].unique())
    x = np.arange(len(qubits))
    width = 0.2

    for i, mtype in enumerate(model_types):
        sub = df[df["Type"] == mtype].sort_values("Qubits")
        if not sub.empty:
            runtimes = sub["Runtime_Seconds"].values
            ax3.bar(x + (i - 1.5) * width, runtimes, width=width,
                    color=colors.get(mtype, "#333"), alpha=0.85,
                    edgecolor="#222", label=mtype)

    ax3.set_xticks(x)
    ax3.set_xticklabels([f"N={q}" for q in qubits])
    ax3.set_yscale("log")
    ax3.set_title("Training Runtime Comparison (Log Scale)", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Number of Qubits (N)", fontsize=11)
    ax3.set_ylabel("Runtime in Seconds (Log Scale)", fontsize=11)
    ax3.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 4: Noise Sensitivity & Degradation Gap
    ax4 = axes[1, 1]
    # Compute the Perplexity difference between Ideal and Noisy models
    q_ideal = df[df["Type"] == "Ideal Quantum"].set_index("Qubits")["Perplexity"]
    q_decoh = df[df["Type"] == "Decoherence"].set_index("Qubits")["Perplexity"]
    q_qpu   = df[df["Type"] == "Physical QPU"].set_index("Qubits")["Perplexity"]
    q_real  = df[df["Type"] == "Real QPU"].set_index("Qubits")["Perplexity"]

    common_q = [q for q in qubits if q in q_ideal.index and q in q_decoh.index and q in q_qpu.index]
    if common_q:
        decoh_gap = [q_decoh.loc[q] - q_ideal.loc[q] for q in common_q]
        qpu_gap   = [q_qpu.loc[q]   - q_ideal.loc[q] for q in common_q]

        bar_w = 0.25
        ax4.bar(np.array(common_q) - bar_w, decoh_gap, width=bar_w,
                color=colors["Decoherence"], alpha=0.85, edgecolor="#222",
                label="Gaussian Decoherence Penalty (+Δ PPL)")
        ax4.bar(np.array(common_q),          qpu_gap,   width=bar_w,
                color=colors["Physical QPU"], alpha=0.85, edgecolor="#222",
                label="IBM Brisbane QPU Noise Penalty (+Δ PPL)")

    # Overlay Real QPU bar at N=10 if Model E results are present
    real_common = [q for q in [10] if q in q_ideal.index and q in q_real.index]
    if real_common:
        real_gap = [q_real.loc[q] - q_ideal.loc[q] for q in real_common]
        bar_w    = 0.25
        ax4.bar(np.array(real_common) + bar_w, real_gap, width=bar_w,
                color=colors["Real QPU"], alpha=0.85, edgecolor="#222",
                label="Real IBM QPU Noise Penalty (+Δ PPL) [Model E]")

        ax4.set_title("Noise Penalty over Ideal VQC (+Δ Perplexity)", fontsize=12, fontweight="bold", pad=10)
        ax4.set_xlabel("Number of Qubits (N)", fontsize=11)
        ax4.set_ylabel("Perplexity Increase (Lower is Better)", fontsize=11)
        ax4.set_xticks(common_q)
        ax4.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
        ax4.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Artifact] 4-Model Qubit scaling figure saved to: {save_path}")


def run_full_scaling_study(qubit_counts=[2, 4, 6, 8, 10], steps=150, lr=0.003):
    print("\n" + "#" * 85)
    print("      COMPREHENSIVE 4-MODEL QUBIT SCALING STUDY (N = 2, 4, 6, 8, 10 QUBITS)      ")
    print("#" * 85)

    set_seed(42)
    data_module = ShakespeareDataModule(seq_len=16, batch_size=16, val_split=0.1, seed=42)
    vocab_size = data_module.vocab_size

    results = []

    for n_qubits in qubit_counts:
        print(f"\n{'#'*85}")
        print(f">>> BENCHMARKING QUBIT COUNT N = {n_qubits} (Hilbert Space Dimension = {2**n_qubits}) <<<")
        print(f"{'#'*85}")

        models = [
            ("Classical", ModelA(vocab_size=vocab_size, bottleneck_dim=n_qubits)),
            ("Ideal Quantum", ModelB(vocab_size=vocab_size, n_qubits=n_qubits)),
            ("Decoherence", ModelC(vocab_size=vocab_size, n_qubits=n_qubits, noise_std=0.05)),
            ("Physical QPU", ModelD(vocab_size=vocab_size, n_qubits=n_qubits, shots=64)),
        ]

        for mtype, model in models:
            set_seed(42)
            metrics = train_model(
                model=model,
                data_module=data_module,
                num_steps=steps,
                lr=lr,
                eval_interval=50,
                log_interval=25
            )
            results.append({
                "Qubits": n_qubits,
                "Hilbert_Dim": 2**n_qubits,
                "Type": mtype,
                "Model": getattr(model, "name", model.__class__.__name__),
                "Train_Loss": metrics["final_train_loss"],
                "Val_Loss": metrics["val_loss"],
                "Perplexity": metrics["perplexity"],
                "Trainable_Params": metrics["trainable_params"],
                "Runtime_Seconds": metrics["runtime_seconds"],
            })

    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("                    4-MODEL QUBIT SCALING EMPIRICAL SUMMARY TABLE")
    print("=" * 90)
    print(df[["Qubits", "Type", "Train_Loss", "Val_Loss", "Perplexity", "Runtime_Seconds"]].to_string(index=False))
    print("=" * 90)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))

    csv_local = os.path.join(script_dir, "qubit_scaling_data.csv")
    csv_root = os.path.join(workspace_root, "qubit_scaling_data.csv")
    df.to_csv(csv_local, index=False)
    df.to_csv(csv_root, index=False)
    print(f"\n[Artifact] Saved 4-model scaling data to:\n  - {csv_local}\n  - {csv_root}")

    # Merge Model E (Real IBM QPU) 10-qubit row if present
    model_e_csv = os.path.join(script_dir, "ibm_model_e_results", "model_e_real_qpu_statistics.csv")
    if os.path.exists(model_e_csv):
        print("\n[Model E] Real IBM QPU results found -- merging into qubit_scaling_data.csv")
        df_e_raw = pd.read_csv(model_e_csv)
        scaling_cols = ["Qubits", "Hilbert_Dim", "Type", "Model", "Train_Loss",
                        "Val_Loss", "Perplexity", "Trainable_Params", "Runtime_Seconds"]
        df_e = df_e_raw[[c for c in scaling_cols if c in df_e_raw.columns]].copy()
        # Remove any pre-existing Real QPU N=10 row to avoid duplicates
        df_base = df[~((df["Type"] == "Real QPU") & (df["Qubits"] == 10))]
        df = pd.concat([df_base, df_e], ignore_index=True)
        df.to_csv(csv_local, index=False)
        df.to_csv(csv_root, index=False)
        print(df[["Qubits", "Type", "Val_Loss", "Perplexity"]].to_string(index=False))
    else:
        print("\n[Model E] No ibm_model_e_results/model_e_real_qpu_statistics.csv found.")
        print("          Run model_e_runner.py to add real IBM QPU results at N=10.")

    png_local = os.path.join(script_dir, "qubit_scaling_analysis.png")
    png_root = os.path.join(workspace_root, "qubit_scaling_analysis.png")
    plot_scaling_analysis(df, png_local)
    shutil.copyfile(png_local, png_root)
    print(f"[Artifact] Saved scaling figure to:\n  - {png_local}\n  - {png_root}")

    return df


if __name__ == "__main__":
    run_full_scaling_study(qubit_counts=[2, 4, 6, 8, 10], steps=150, lr=0.003)
