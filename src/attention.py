"""attention.py — Causal (masked) multi-head self-attention.

This is the core computation inside every Transformer block:
  1. Project input x into queries, keys, values via a single fused linear layer.
  2. Split into n_head independent heads.
  3. Compute scaled dot-product attention with causal masking.
  4. Concatenate heads and project back to model dimension.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal (left-to-right) masking.

    Args:
        n_embd   : total embedding / model dimension (C)
        n_head   : number of attention heads
        dropout  : attention dropout probability during training

    Shapes flowing through forward():
        input  x : (B, T, C)   — batch, sequence length, embedding dim
        output y : (B, T, C)   — same shape, attended context
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.d_head = n_embd // n_head   # dimension per head

        # Single fused projection for Q, K, V — more efficient than 3 separate linears
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection after concatenating all heads
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        B, T, C = x.shape

        # --- 1. Compute Q, K, V for all heads in one matrix multiply ---
        # qkv: (B, T, 3*C) → view as (B, T, 3, n_head, d_head)
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, n_head, d_head)

        # Reshape to (B, n_head, T, d_head) — PyTorch SDPA expects this layout
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # --- 2. Scaled dot-product attention with causal mask ---
        # is_causal=True applies the upper-triangular mask automatically.
        # When flash attention is available, PyTorch routes here for free.
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,   # prevent attending to future positions
        )

        # --- 3. Reassemble heads: (B, n_head, T, d_head) → (B, T, C) ---
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --- 4. Output projection ---
        y = self.proj(y)
        return y
