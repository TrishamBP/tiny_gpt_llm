"""model.py — Full GPT-style language model.

Layers stacked in order:
  tok_emb + pos_emb → dropout → [Block × n_layer] → LayerNorm → linear head

Each Block = LayerNorm → CausalSelfAttention (residual) → LayerNorm → FeedForward (residual)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .utils import top_k_top_p_filtering


# ---------------------------------------------------------------------------
# Feed-Forward Network (MLP inside each transformer block)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise two-layer MLP with GELU activation.

    Expands the embedding dimension by `mult` (default 4×), applies GELU,
    then projects back down. This is the "FFN" in the GPT paper.

    Shapes: (B, T, n_embd) → (B, T, n_embd)
    """

    def __init__(self, n_embd: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """One transformer block: Pre-LN attention + Pre-LN FFN, both with residuals.

    Pre-LayerNorm order (LN before sublayer, not after) matches GPT-2 and
    most modern implementations — it stabilizes training at depth.

    Forward:
        x = x + attn(ln1(x))   # attend, add residual
        x = x + ffn(ln2(x))    # mix, add residual
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # attention sub-layer
        x = x + self.ffn(self.ln2(x))    # feed-forward sub-layer
        return x


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Tiny GPT-style autoregressive language model.

    Args:
        vocab_size  : number of token types (256 for byte-level)
        block_size  : maximum context window / sequence length
        n_layer     : number of stacked transformer blocks
        n_head      : attention heads per block
        n_embd      : model dimension (embedding size)
        dropout     : dropout applied to embeddings and attention

    Forward signature:
        logits, loss = model(idx, targets)
        logits, _    = model(idx)          # inference — no loss
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size

        # Token + position embeddings (both learned)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])

        # Final LayerNorm before the output head
        self.ln_f = nn.LayerNorm(n_embd)

        # Linear projection from model dim → vocabulary logits (no bias, following GPT-2)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight initialisation: N(0, 0.02) for linears and embeddings (GPT-2 convention)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,                  # (B, T) token IDs
        targets: torch.Tensor | None = None,  # (B, T) next-token labels
    ):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        # Absolute position indices [0, 1, ..., T-1] broadcast over batch
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)

        # Embed tokens and positions, then add them
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        x = self.drop(x)

        # Pass through all transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)                           # final norm
        logits = self.head(x)                      # (B, T, vocab_size)

        # Compute cross-entropy loss if training targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                  # (B*T,)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,           # (B, T) prompt token IDs
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens one at a time.

        At each step:
          1. Crop context to block_size
          2. Forward pass → logits for last position
          3. Scale by temperature, apply top-k/top-p filter
          4. Sample from resulting distribution
          5. Append sampled token and repeat
        """
        self.eval()

        # Seed with a newline byte if prompt is empty
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)

        for _ in range(max_new_tokens):
            # Crop to context window (never exceed block_size)
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # (B, vocab)

            # Filter logit distribution
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)              # grow sequence

        return idx
