"""config.py — Default hyperparameters for Tiny GPT.

All values can be overridden via train.py's CLI arguments.
They live here as a single source of truth for the defaults.
"""
from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    # ---- Architecture ----
    vocab_size: int   = 256     # byte-level tokenizer: always 256
    block_size: int   = 256     # context window (max sequence length)
    n_layer:    int   = 4       # number of transformer blocks
    n_head:     int   = 4       # attention heads per block
    n_embd:     int   = 256     # model / embedding dimension
    dropout:    float = 0.0     # dropout probability (0 = disabled)

    # ---- Training ----
    batch_size:    int   = 32
    steps:         int   = 2000
    lr:            float = 3e-4
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0

    # ---- Evaluation & sampling ----
    eval_interval: int   = 200
    eval_iters:    int   = 50
    sample_every:  int   = 200
    sample_tokens: int   = 256
    temperature:   float = 1.0
    top_k:         int   = 50
    top_p:         float = None  # type: ignore  # None disables nucleus sampling

    # ---- I/O ----
    out_dir: str = 'runs/tiny-gpt'


# Convenience: a small, fast config for smoke-testing (fits on CPU in minutes)
SMOKE_CONFIG = GPTConfig(
    block_size=128,
    n_layer=2,
    n_head=2,
    n_embd=128,
    steps=400,
    eval_interval=100,
    sample_every=100,
    batch_size=32,
)
