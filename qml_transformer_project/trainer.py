"""
Training and evaluation engine for Quantum and Classical Transformer models.
Provides unified training loops, validation routines, metric collection,
perplexity computation, and model checkpointing.
"""

import os
import re
import time
import math
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
from dataset import ShakespeareDataModule

# Shared checkpoint directory (sibling to this script)
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _safe_filename(name: str) -> str:
    """Converts a model name string to a safe filename stem."""
    return re.sub(r"[^\w\-]", "_", name).strip("_")


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model: nn.Module, data_module: ShakespeareDataModule, eval_steps: int = 10) -> float:
    """
    Evaluates model across multiple validation batches and computes mean validation loss.
    """
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = data_module.get_batch(split="val")
            _, loss = model(x, y)
            total_val_loss += loss.item()
    return total_val_loss / eval_steps


def train_model(
    model: nn.Module,
    data_module: ShakespeareDataModule,
    num_steps: int = 150,
    lr: float = 0.003,
    eval_interval: int = 25,
    log_interval: int = 10,
) -> Dict[str, Any]:
    """
    Trains a model for num_steps using AdamW optimizer and tracks thesis metrics.

    Returns:
        Dictionary containing:
          - 'model_name': Name identifier
          - 'trainable_params': Number of trainable parameters
          - 'runtime_seconds': Execution duration in seconds
          - 'final_train_loss': Final training loss
          - 'val_loss': Final validation loss
          - 'perplexity': exp(val_loss)
          - 'history': Step-by-step training and validation loss history
    """
    model_name = getattr(model, "name", model.__class__.__name__)
    print(f"\n{'='*70}")
    print(f"Starting Training: {model_name}")
    print(f"Trainable Parameters: {count_trainable_parameters(model):,}")
    print(f"Optimization: AdamW (lr={lr}, steps={num_steps})")
    print(f"{'='*70}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    train_loss_history: List[float] = []
    val_loss_history: List[Dict[str, float]] = []
    
    start_time = time.time()
    
    for step in range(1, num_steps + 1):
        model.train()
        x, y = data_module.get_batch(split="train")

        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        train_loss_history.append(loss_val)

        if step % log_interval == 0 or step == 1 or step == num_steps:
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / step) * 1000
            print(f"  Step [{step:3d}/{num_steps:3d}] | Loss: {loss_val:.4f} | {ms_per_step:.1f} ms/step | Elapsed: {elapsed:.1f}s")

        if step % eval_interval == 0 or step == num_steps:
            v_loss = evaluate_model(model, data_module, eval_steps=5)
            val_loss_history.append({"step": step, "val_loss": v_loss})
            print(f"  --> [Validation @ Step {step}] Loss: {v_loss:.4f} | Perplexity: {math.exp(v_loss):.2f}")

    total_runtime = time.time() - start_time
    final_train_loss = train_loss_history[-1]
    final_val_loss = evaluate_model(model, data_module, eval_steps=10)
    perplexity = math.exp(final_val_loss)

    print(f"\n[Finished {model_name}]")
    print(f"  Total Runtime: {total_runtime:.2f}s")
    print(f"  Final Train Loss: {final_train_loss:.4f}")
    print(f"  Final Val Loss:   {final_val_loss:.4f}")
    print(f"  Perplexity (PPL): {perplexity:.2f}")

    # ── Checkpoint: save state_dict so generate.py can reload any model ──
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_stem = _safe_filename(model_name)
    ckpt_path = CHECKPOINT_DIR / f"{ckpt_stem}.pt"
    torch.save(
        {
            "model_class": model.__class__.__name__,
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "vocab_size": data_module.vocab_size,
            "val_loss": round(final_val_loss, 4),
            "perplexity": round(perplexity, 4),
        },
        ckpt_path,
    )
    print(f"  [Checkpoint] Saved -> {ckpt_path}")

    return {
        "model_name": model_name,
        "trainable_params": count_trainable_parameters(model),
        "runtime_seconds": round(total_runtime, 2),
        "final_train_loss": round(final_train_loss, 4),
        "val_loss": round(final_val_loss, 4),
        "perplexity": round(perplexity, 4),
        "checkpoint": str(ckpt_path),
        "history": {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history
        }
    }
