"""dataset.py — Byte-level dataset and batching.

Reads an entire text file into memory as a flat tensor of byte values (0-255).
Splits into train / val, then serves random (x, y) pairs where:
  - x[i] = bytes at positions [start .. start+block_size)
  - y[i] = bytes at positions [start+1 .. start+block_size+1)  ← one-step shift

This shifted-by-one construction is the standard language-model objective:
predict the next byte given all preceding bytes in the window.
"""
from __future__ import annotations
from pathlib import Path
import torch


class ByteDataset:
    """Holds raw bytes of a text file and serves (x, y) blocks for LM training.

    Args:
        path       : path to any text file (UTF-8 or raw bytes)
        block_size : context window length (T in the model)
        split      : fraction of data used for training; remainder is validation
    """

    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = Path(path).read_bytes()
        # Each byte becomes one integer token in [0, 255]
        data = torch.tensor(list(data), dtype=torch.long)

        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(
        self,
        which: str,          # 'train' or 'val'
        batch_size: int,
        device: torch.device,
    ):
        """Sample a random batch of (input, target) token sequences.

        Returns:
            x : (batch_size, block_size) — input tokens
            y : (batch_size, block_size) — target tokens (x shifted right by 1)
        """
        buf = self.train if which == 'train' else self.val
        assert len(buf) > self.block_size + 1, 'File too small for the given block_size'

        # Random starting positions for each sample in the batch
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        x = torch.stack([buf[i:i + self.block_size] for i in ix])
        y = torch.stack([buf[i + 1:i + 1 + self.block_size] for i in ix])

        return x.to(device), y.to(device)
