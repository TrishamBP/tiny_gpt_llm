"""utils.py — Tokenizer and sampling helpers.

Two responsibilities:
  1. ByteTokenizer  : convert text ↔ integer token IDs (vocab_size = 256)
  2. top_k_top_p_filtering : truncate a logit distribution before sampling
"""
from __future__ import annotations
import torch


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class ByteTokenizer:
    """Ultra-simple byte-level tokenizer.

    Every byte (0-255) is a token — no learned merges, no special tokens.
    This means vocab_size is always 256, regardless of language or domain.

    Usage:
        tok = ByteTokenizer()
        ids = tok.encode("hello")   # LongTensor of byte values
        txt = tok.decode(ids)       # back to string
    """

    def encode(self, s: str) -> torch.Tensor:
        """UTF-8 encode a string, return a LongTensor of byte values."""
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        """Convert a sequence of byte IDs back to a UTF-8 string.
        Silently drops bytes that form invalid UTF-8 sequences.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Filter logits to retain only the top-k / nucleus (top-p) probability mass.

    Any token outside the selected set is set to -inf so that softmax assigns
    it probability 0.

    Args:
        logits : (B, vocab_size) raw model output for the current step
        top_k  : keep only the k tokens with the highest logit
        top_p  : keep the smallest set of tokens whose cumulative probability
                 exceeds p (nucleus sampling)

    Returns:
        filtered logits (B, vocab_size) — same shape, masked entries = -inf
    """
    B, V = logits.shape
    filtered = logits.clone()

    # Top-k: zero out everything below the k-th largest logit
    if top_k is not None and top_k < V:
        topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
        kth = topk_vals[:, -1].unsqueeze(-1)   # value of the k-th token
        filtered[filtered < kth] = float('-inf')

    # Nucleus (top-p): keep the smallest set of tokens whose cumulative
    # probability exceeds p when sorted by descending probability
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)

        # Mask tokens AFTER the nucleus boundary (keep at least 1 token)
        mask = cumsum > top_p
        mask[..., 0] = False                   # always keep the top-1 token
        sorted_logits[mask] = float('-inf')

        # Scatter masked logits back to original token ordering
        filtered = torch.full_like(filtered, float('-inf'))
        filtered.scatter_(1, sorted_idx, sorted_logits)

    return filtered
