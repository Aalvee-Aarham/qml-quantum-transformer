"""
Dataset pipeline and character-level tokenizer for Tiny Shakespeare.
Automatically retrieves 20,000 characters, builds vocabulary mappings,
and provisions training/validation mini-batches for autoregressive next-token prediction.
"""

import os
import urllib.request
from typing import Tuple, List
import torch

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE_PATH = os.path.join(DATA_CACHE_DIR, "tinyshakespeare_20k.txt")
MAX_CHARS = 20000


def download_data(url: str = SHAKESPEARE_URL, cache_path: str = DATA_FILE_PATH, max_chars: int = MAX_CHARS) -> str:
    """
    Downloads or reads cached Tiny Shakespeare text up to max_chars characters.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        print(f"[Dataset] Downloading {max_chars} characters from {url}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_text = response.read().decode("utf-8")
        except Exception as e:
            print(f"[Dataset] Download failed ({e}), using fallback Shakespeare corpus.")
            raw_text = (
                "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
                "All:\nSpeak, speak.\n\n"
                "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
                "All:\nResolved. resolved.\n\n"
                "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"
            ) * 300
        text = raw_text[:max_chars]
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(cache_path, "r", encoding="utf-8") as f:
            text = f.read()[:max_chars]
    
    print(f"[Dataset] Loaded {len(text)} characters from {cache_path}")
    return text


class CharTokenizer:
    """
    Character-level tokenizer for text encoding and decoding.
    """
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, string: str) -> List[int]:
        return [self.stoi[c] for c in string if c in self.stoi]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.itos[i] for i in indices if i in self.itos])


class ShakespeareDataModule:
    """
    Manages encoded text, train/validation split, and batch generation.
    """
    def __init__(self, seq_len: int = 16, batch_size: int = 16, val_split: float = 0.1, seed: int = 42):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)

        self.raw_text = download_data()
        self.tokenizer = CharTokenizer(self.raw_text)
        self.vocab_size = self.tokenizer.vocab_size

        encoded_data = torch.tensor(self.tokenizer.encode(self.raw_text), dtype=torch.long)
        n = int((1.0 - val_split) * len(encoded_data))
        self.train_data = encoded_data[:n]
        self.val_data = encoded_data[n:]
        print(f"[Dataset] Vocab size: {self.vocab_size} | Train tokens: {len(self.train_data)} | Val tokens: {len(self.val_data)}")

    def get_batch(self, split: str = "train") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a causal next-token batch (X, Y) where Y is shifted by 1 relative to X.
        X shape: (batch_size, seq_len)
        Y shape: (batch_size, seq_len)
        """
        data = self.train_data if split == "train" else self.val_data
        max_idx = len(data) - self.seq_len - 1
        ix = torch.randint(0, max_idx, (self.batch_size,), generator=self.rng)
        
        x = torch.stack([data[i : i + self.seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.seq_len] for i in ix])
        return x, y
