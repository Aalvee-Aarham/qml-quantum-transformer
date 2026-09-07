"""
Main orchestration script for Quantum-Classical Hybrid Transformer Comparative Study.
Executes training for:
  - Model A: Classical Baseline
  - Model B: Ideal PennyLane 10-qubit VQC
  - Model C: Decoherence-Simulated 10-qubit VQC (std=0.05)
  - Model D: Physical QPU-Simulated 10-qubit VQC (Qiskit Aer FakeBrisbane)
Exports:
  - thesis_data_collection.csv
  - thesis_loss_curves.png
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
from models import ModelA, ModelB, ModelC, ModelD
from trainer import train_model


def set_seed(seed: int = 42):
    """Sets random seeds for full experiment reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_thesis_curves(results: list, save_path: str):
    """
    Generates a publication-quality 4-panel figure comparing training/validation dynamics,
    perplexity, and runtime efficiency across all models.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor("#fafbfc")

    colors = {
        "Model A": "#1f77b4",  # Classical Blue
        "Model B": "#2ca02c",  # Quantum Green
        "Model C": "#ff7f0e",  # Decoherence Orange
        "Model D": "#9467bd",  # Hardware Violet
    }

    # Helper label resolver
    def get_short_name(full_name: str) -> str:
        for k in colors:
            if k in full_name:
                return k
        return full_name[:7]

    # Panel 1: Step-wise Training Loss Convergence
    ax1 = axes[0, 0]
    for r in results:
        label = get_short_name(r["model_name"])
        color = colors.get(label, "#333333")
        losses = r["history"]["train_loss"]
        steps = list(range(1, len(losses) + 1))
        
        # Raw loss with alpha
        ax1.plot(steps, losses, color=color, alpha=0.25, linewidth=1.0)
        # Moving average for smooth publication-quality presentation
        window = 10
        if len(losses) >= window:
            smooth_loss = pd.Series(losses).rolling(window, min_periods=1).mean()
            ax1.plot(steps, smooth_loss, color=color, linewidth=2.2, label=f"{label} (smoothed)")
        else:
            ax1.plot(steps, losses, color=color, linewidth=2.0, label=label)

    ax1.set_title("Training Loss Convergence (Cross-Entropy)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Training Step", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Loss Progression
    ax2 = axes[0, 1]
    for r in results:
        label = get_short_name(r["model_name"])
        color = colors.get(label, "#333333")
        val_records = r["history"]["val_loss"]
        val_steps = [v["step"] for v in val_records]
        val_losses = [v["val_loss"] for v in val_records]
        ax2.plot(val_steps, val_losses, marker="o", markersize=6, linewidth=2.0, color=color, label=label)

    ax2.set_title("Validation Loss Checkpoints", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Training Step", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Final Validation Perplexity (PPL) Bar Chart
    ax3 = axes[1, 0]
    names = [get_short_name(r["model_name"]) for r in results]
    ppls = [r["perplexity"] for r in results]
    bar_colors = [colors.get(n, "#555555") for n in names]
    
    bars = ax3.bar(names, ppls, color=bar_colors, edgecolor="#222222", width=0.55, alpha=0.88)
    for bar in bars:
        h = bar.get_height()
        ax3.annotate(f"{h:.2f}",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax3.set_title("Validation Perplexity Comparison (exp(Val Loss))", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    ax3.grid(axis="y", linestyle="--", alpha=0.6)

    # Panel 4: Runtime & Parameter Trade-Off
    ax4 = axes[1, 1]
    runtimes = [r["runtime_seconds"] for r in results]
    params = [r["trainable_params"] for r in results]

    b_runtime = ax4.bar(np.arange(len(names)), runtimes, width=0.5, color=bar_colors, edgecolor="#222222", alpha=0.85)
    for idx, bar in enumerate(b_runtime):
        h = bar.get_height()
        ax4.annotate(f"{h:.1f}s\n({params[idx]:,}p)",
                     xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax4.set_xticks(np.arange(len(names)))
    ax4.set_xticklabels(names)
    ax4.set_title("Total Training Runtime & Parameter Efficiency", fontsize=13, fontweight="bold", pad=10)
    ax4.set_ylabel("Runtime (Seconds)", fontsize=11)
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout(pad=3.0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[Artifact] High-resolution convergence graph saved to: {save_path}")


def main():
    print("\n" + "#" * 80)
    print("  QUANTUM-CLASSICAL TRANSFORMER BENCHMARK: THESIS EXPERIMENTAL SUITE  ")
    print("#" * 80)

    # 1. Reproducibility seed & Data Loading
    set_seed(42)
    data_module = ShakespeareDataModule(seq_len=16, batch_size=16, val_split=0.1, seed=42)
    vocab_size = data_module.vocab_size

    # 2. Instantiate all 4 model architectures
    print("\n[Architecture] Instantiating 4 Model Variants (10 Qubits, Depth=2):")
    model_a = ModelA(vocab_size=vocab_size)
    model_b = ModelB(vocab_size=vocab_size)
    model_c = ModelC(vocab_size=vocab_size, noise_std=0.05)
    model_d = ModelD(vocab_size=vocab_size, shots=128)

    models_to_evaluate = [model_a, model_b, model_c, model_d]

    # 3. Train all models sequentially
    results = []
    num_steps = 150
    lr = 0.003

    for model in models_to_evaluate:
        # Reset seed before each model training for fair comparison across data batches
        set_seed(42)
        metrics = train_model(
            model=model,
            data_module=data_module,
            num_steps=num_steps,
            lr=lr,
            eval_interval=25,
            log_interval=15
        )
        results.append(metrics)

    # 4. Formulate Summary Table & Export CSV
    summary_data = []
    for r in results:
        summary_data.append({
            "Model": r["model_name"],
            "Train_Loss": r["final_train_loss"],
            "Val_Loss": r["val_loss"],
            "Perplexity": r["perplexity"],
            "Trainable_Params": r["trainable_params"],
            "Runtime_Seconds": r["runtime_seconds"],
        })
    df_summary = pd.DataFrame(summary_data)

    print("\n" + "=" * 80)
    print("                       EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80)

    # File paths (both in project directory and workspace root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))

    csv_local = os.path.join(script_dir, "thesis_data_collection.csv")
    csv_root = os.path.join(workspace_root, "thesis_data_collection.csv")
    df_summary.to_csv(csv_local, index=False)
    df_summary.to_csv(csv_root, index=False)
    print(f"[Artifact] Exported quantitative metrics to:\n  - {csv_local}\n  - {csv_root}")

    # 5. Generate and export publication-grade graph
    png_local = os.path.join(script_dir, "thesis_loss_curves.png")
    png_root = os.path.join(workspace_root, "thesis_loss_curves.png")
    plot_thesis_curves(results, png_local)
    shutil.copyfile(png_local, png_root)
    print(f"[Artifact] Exported publication plot to:\n  - {png_local}\n  - {png_root}")

    print("\n[Success] All 10-qubit quantum simulations and evaluations completed flawlessly!")


if __name__ == "__main__":
    main()
