"""
generate.py — Autoregressive Text Generation from Saved Checkpoints.

Usage:
    python generate.py                          # auto-selects best checkpoint
    python generate.py --model checkpoints/Model_A__Classical_Baseline___Dim_10_.pt
    python generate.py --prompt "To be or" --max_new_tokens 200 --temperature 0.8

All 4 models (A, B, C, D) share the same checkpoint format produced by trainer.py.
The script reconstructs the exact architecture from the saved model_class name.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

# Ensure project package is importable when run from any directory
PROJECT_DIR = Path(__file__).parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from dataset import ShakespeareDataModule
from models import ModelA, ModelB, ModelC, ModelD


# ── Model registry: maps class name → constructor ────────────────────────────
MODEL_REGISTRY = {
    "ModelA": lambda vocab_size, **_: ModelA(vocab_size=vocab_size),
    "ModelB": lambda vocab_size, **_: ModelB(vocab_size=vocab_size),
    "ModelC": lambda vocab_size, **_: ModelC(vocab_size=vocab_size, noise_std=0.05),
    "ModelD": lambda vocab_size, **_: ModelD(vocab_size=vocab_size, shots=128),
}


# ── Sampling helpers ──────────────────────────────────────────────────────────

def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits except the top-k, preventing unlikely tokens."""
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return logits.masked_fill(logits < min_values, float("-inf"))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 40,
) -> torch.Tensor:
    """
    Autoregressive token generation.

    Args:
        model:          Trained model with forward(idx) -> (logits, loss)
        idx:            (1, T) LongTensor of seed token indices
        max_new_tokens: Number of new tokens to generate
        temperature:    Sampling temperature (lower = more deterministic)
        top_k:          Restrict sampling to top-k most probable tokens

    Returns:
        (1, T + max_new_tokens) LongTensor of token indices
    """
    model.eval()
    seq_len = 16  # matches training seq_len in ShakespeareDataModule

    for _ in range(max_new_tokens):
        # Crop context to the model's maximum sequence length
        idx_cond = idx[:, -seq_len:]

        # Pad with zeros if shorter than seq_len (beginning of generation)
        if idx_cond.shape[1] < seq_len:
            pad = torch.zeros(1, seq_len - idx_cond.shape[1], dtype=torch.long)
            idx_cond = torch.cat([pad, idx_cond], dim=1)

        logits, _ = model(idx_cond)          # (1, T, vocab_size)
        logits = logits[:, -1, :]            # last time-step -> (1, vocab_size)
        logits = logits / temperature
        logits = top_k_logits(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        idx = torch.cat([idx, next_token], dim=1)

    return idx


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def list_checkpoints(ckpt_dir: Path):
    """Returns all .pt files in the checkpoints directory, sorted."""
    return sorted(ckpt_dir.glob("*.pt"))


def pick_best_checkpoint(ckpt_dir: Path) -> Path:
    """
    Automatically selects the checkpoint with the lowest validation loss.
    Falls back to the first available checkpoint if metadata is missing.
    """
    candidates = list_checkpoints(ckpt_dir)
    if not candidates:
        raise FileNotFoundError(
            "No .pt checkpoints found in '{}'.\n"
            "Run main.py first to train the models and generate checkpoints.".format(ckpt_dir)
        )

    best_path, best_loss = None, float("inf")
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            loss = ckpt.get("val_loss", float("inf"))
            if loss < best_loss:
                best_loss, best_path = loss, path
        except Exception:
            continue

    return best_path or candidates[0]


def load_model_from_checkpoint(ckpt_path: Path):
    """
    Loads a checkpoint and reconstructs the model.

    Returns:
        (model, checkpoint_dict, data_module)
    """
    print("\n[Load] Reading checkpoint: {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model_class = ckpt["model_class"]
    model_name  = ckpt["model_name"]
    vocab_size  = ckpt["vocab_size"]
    val_loss    = ckpt.get("val_loss", "N/A")
    perplexity  = ckpt.get("perplexity", "N/A")

    print("  Model Class : {}".format(model_class))
    print("  Model Name  : {}".format(model_name))
    print("  Vocab Size  : {}".format(vocab_size))
    print("  Val Loss    : {}".format(val_loss))
    print("  Perplexity  : {}".format(perplexity))

    if model_class not in MODEL_REGISTRY:
        raise ValueError(
            "Unknown model class '{}'. Available: {}".format(
                model_class, list(MODEL_REGISTRY.keys())
            )
        )

    model = MODEL_REGISTRY[model_class](vocab_size=vocab_size)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Build data module only to get the shared tokenizer (stoi/itos maps)
    data_module = ShakespeareDataModule(seq_len=16, batch_size=1, val_split=0.1, seed=42)

    return model, ckpt, data_module


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from a trained quantum/classical transformer checkpoint."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a specific .pt checkpoint. Defaults to the best available checkpoint."
    )
    parser.add_argument(
        "--prompt", type=str, default="ROMEO:",
        help="Seed text for generation (default: 'ROMEO:')"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=150,
        help="Number of tokens to generate (default: 150)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8; lower = more focused)"
    )
    parser.add_argument(
        "--top_k", type=int, default=40,
        help="Top-k sampling cutoff (default: 40)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available checkpoints and exit."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ckpt_dir = PROJECT_DIR / "checkpoints"

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list:
        checkpoints = list_checkpoints(ckpt_dir)
        if not checkpoints:
            print("No checkpoints found. Run main.py first.")
            return
        print("\nAvailable checkpoints in '{}':\n".format(ckpt_dir))
        for p in checkpoints:
            try:
                c = torch.load(p, map_location="cpu", weights_only=False)
                print("  {:<55}  val_loss={:.4f}  ppl={:.2f}  [{}]".format(
                    p.name,
                    c.get("val_loss", 0.0),
                    c.get("perplexity", 0.0),
                    c.get("model_class", "?")
                ))
            except Exception as e:
                print("  {}  (could not read: {})".format(p.name, e))
        return

    # ── Select checkpoint ─────────────────────────────────────────────────────
    if args.model:
        ckpt_path = Path(args.model)
        if not ckpt_path.exists():
            ckpt_path = PROJECT_DIR / args.model
        if not ckpt_path.exists():
            raise FileNotFoundError("Checkpoint not found: {}".format(args.model))
    else:
        print("[Auto-select] Finding best checkpoint by lowest val_loss ...")
        ckpt_path = pick_best_checkpoint(ckpt_dir)
        print("  Selected: {}".format(ckpt_path.name))

    # ── Load model ────────────────────────────────────────────────────────────
    model, ckpt, data_module = load_model_from_checkpoint(ckpt_path)

    # ── Tokenize prompt ───────────────────────────────────────────────────────
    stoi = data_module.stoi  # char -> int
    itos = data_module.itos  # int  -> char

    prompt = args.prompt
    unknown_chars = [c for c in prompt if c not in stoi]
    if unknown_chars:
        print("\n[Warning] Prompt contains unknown characters {}, they will be skipped.".format(unknown_chars))
        prompt = "".join(c for c in prompt if c in stoi)
    if not prompt:
        prompt = " "

    seed_ids = [stoi[c] for c in prompt]
    idx = torch.tensor([seed_ids], dtype=torch.long)  # (1, len(prompt))

    # ── Generate ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Model     : {}".format(ckpt["model_name"]))
    print("  Prompt    : {}".format(repr(prompt)))
    print("  New tokens: {}".format(args.max_new_tokens))
    print("  Temp      : {}  |  Top-k: {}".format(args.temperature, args.top_k))
    print("=" * 60 + "\n")

    out_idx = generate(
        model=model,
        idx=idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # Decode all tokens (prompt + generated)
    generated_ids = out_idx[0].tolist()
    generated_text = "".join(itos.get(i, "?") for i in generated_ids)

    print(generated_text)
    print("\n" + "=" * 60)
    print("[Done] {} tokens generated from {}".format(args.max_new_tokens, ckpt_path.name))


if __name__ == "__main__":
    main()
