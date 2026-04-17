# Tiny LLM — From Scratch (GPT-style)

A minimal, fully working GPT-style language model built from first principles.
~3 million parameters. Trains on a laptop CPU in minutes. Zero external ML libraries beyond PyTorch.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Big Picture — What Are We Actually Building?](#2-big-picture)
3. [Architecture Breakdown](#3-architecture-breakdown)
   - 3.1 [Tokenization & Dataset](#31-tokenization--dataset)
   - 3.2 [Embeddings](#32-embeddings)
   - 3.3 [Self-Attention](#33-self-attention)
   - 3.3.1 [Causal Masking](#331-causal-masking)
   - 3.4 [Multi-Head Attention](#34-multi-head-attention)
   - 3.5 [Transformer Block](#35-transformer-block)
   - 3.6 [Output Layer](#36-output-layer)
4. [Training Process](#4-training-process)
5. [Key Differences vs GPT-2 / GPT-3](#5-key-differences-vs-gpt)
6. [Step-by-Step Data Flow](#6-step-by-step-data-flow)
7. [Code ↔ Concept Mapping](#7-code--concept-mapping)
8. [How to Run](#8-how-to-run)
9. [How to Extend This](#9-how-to-extend-this)

---

## 1. Overview

### What this project does

This project trains a **character-level language model** on any text file. Given a chunk of text
as context, the model learns to predict the next character (more precisely, the next byte).
After training, you can prompt it with any text and it will generate plausible continuations
in the same style as the training data.

### What "tiny LLM" means

| Property      | This model           |
| ------------- | -------------------- |
| Parameters    | ~3M (default)        |
| Vocab         | 256 (all bytes)      |
| Context       | 256 tokens           |
| Layers        | 4 transformer blocks |
| Training data | ~270KB text file     |

It is the minimum viable implementation of a GPT that still trains and generates coherent text.
No clever tricks, no abstractions — just the bare architecture.

### How it relates to GPT

GPT-2 and GPT-3 follow the **exact same architecture** as this model:
embedding → stacked transformer blocks → linear output head.
The only differences are scale (more layers, wider dimensions, more data) and a few engineering
choices around tokenization and weight initialization. Understanding this model means
understanding the core of every large language model in production today.

---

## 2. Big Picture

### What is a language model?

A language model assigns a probability to every possible next token given a sequence of
prior tokens. Formally:

```
P(token_t | token_0, token_1, ..., token_{t-1})
```

During **training** we give it a sequence and ask: "for each position, how confident were you
about the actual next token?" That confidence is the loss. We minimize the loss via gradient descent.

During **inference** we repeatedly sample from that distribution to generate text one token at a time.

### What are we building?

A **decoder-only transformer** (the architecture used by GPT, LLaMA, Mistral, etc.).
"Decoder-only" means:

- Each token can only attend to tokens **before it** (causal masking)
- No encoder, no cross-attention — just autoregressive generation
- Perfectly suited for language generation

### The pipeline at a glance

```
Raw text file
     │
     ▼
[ByteTokenizer]            Convert each byte to an integer in [0, 255]
     │
     ▼
[ByteDataset]              Load into memory, split train/val, serve random batches
     │
     │  x: (B, T)  — input token IDs
     │  y: (B, T)  — target token IDs (x shifted right by 1)
     ▼
[GPT.forward(x, y)]
     │
     ├── tok_emb(x) + pos_emb(pos)     →  (B, T, C)  embeddings
     │
     ├── Block 1 → Block 2 → ... → Block N            transformer stack
     │
     ├── LayerNorm → Linear head        →  (B, T, vocab_size)  logits
     │
     └── cross_entropy(logits, y)       →  scalar loss
          │
          ▼
     [Optimizer]          loss.backward() → update weights → repeat
```

---

## 3. Architecture Breakdown

### 3.1 Tokenization & Dataset

**The intuition:** Before a neural network can process text, text must become numbers.
The simplest possible approach: treat every byte of the file as a token.

**What happens:**

Every character in a UTF-8 text file is already stored as 1–4 bytes. We read those bytes
directly as integers in the range `[0, 255]`. No learned vocabulary, no merging of common pairs
(BPE), no lookup tables — just raw bytes.

```python
# src/utils.py
class ByteTokenizer:
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256
```

**Why this works:** Every possible byte value is a valid token. The model has a fixed vocabulary
of 256 tokens, and nothing about a UTF-8 text file can fall outside that range.
The cost is efficiency: multi-byte characters (like `ä`, `你`, `क`) become 2–4 tokens instead of 1.

**The dataset — next-token prediction setup:**

The fundamental structure of language model training is the _shifted input/target pair_:

```
text  :  H  e  l  l  o
ids   :  72 101 108 108 111

x     :  [72, 101, 108, 108]   ← input
y     :  [101, 108, 108, 111]  ← target (x shifted right by 1)
```

At position 0: given `H`, predict `e`.
At position 1: given `H e`, predict `l`.
...and so on. One sequence gives `block_size` training examples.

```python
# src/dataset.py
def get_batch(self, which, batch_size, device):
    ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
    x = torch.stack([buf[i:i + self.block_size] for i in ix])
    y = torch.stack([buf[i + 1:i + 1 + self.block_size] for i in ix])
    return x.to(device), y.to(device)
```

**How GPT does it:** GPT uses Byte Pair Encoding (BPE) via tiktoken, with a vocabulary of ~50,257
tokens. Frequent subword sequences like `" the"` or `" model"` become single tokens. This is
more efficient (shorter sequences for the same text), but the architecture that processes those
tokens is identical.

---

### 3.2 Embeddings

**The intuition:** Token IDs are arbitrary integers. The model needs to convert them into
vectors it can reason about. That's the embedding table.

**Token embeddings:**

An `nn.Embedding(vocab_size, n_embd)` is simply a lookup table of shape `(256, 256)`.
Each row is a learned vector for one token. Index into it with a token ID, get a 256-dimensional vector.

```python
# src/model.py
self.tok_emb = nn.Embedding(vocab_size, n_embd)  # (256, 256)
```

**Positional embeddings:**

Attention has no notion of order — `[A, B, C]` and `[C, A, B]` look the same.
We need to inject position information. The simplest approach: another learned embedding table,
one row per position.

```python
self.pos_emb = nn.Embedding(block_size, n_embd)  # (256, 256)
```

In the forward pass:

```python
pos = torch.arange(0, T, device=idx.device).unsqueeze(0)  # (1, T)
x = self.tok_emb(idx) + self.pos_emb(pos)                 # (B, T, C)
```

Token embeddings and position embeddings are **added together** into a single combined
representation. The model learns to encode position into the same space as token meaning.

**How GPT does it:** GPT-2 also uses learned positional embeddings. GPT-3 switches to
RoPE (Rotary Position Embedding) which generalizes better to sequences longer than those seen
during training — but the learned approach here is conceptually identical.

---

### 3.3 Self-Attention

**The intuition:** For each token in the sequence, self-attention asks:
_"Which other tokens should I pay attention to when building my representation?"_

It produces a weighted average of value vectors, where the weights come from compatibility
between queries and keys.

**The mechanics — Q, K, V:**

Each token's embedding `x_i` is projected into three vectors:

- **Query (Q):** "What am I looking for?"
- **Key (K):** "What do I contain / offer?"
- **Value (V):** "What do I contribute to tokens that attend to me?"

```
attention_score(i, j) = dot(Q_i, K_j) / sqrt(d_head)
```

The division by `sqrt(d_head)` prevents the dot products from growing so large that softmax
saturates into near-one-hot distributions (which would kill gradients).

After softmax over j:

```
output_i = sum_j  softmax(scores)[i,j] * V_j
```

**Implementation — fused QKV projection:**

Rather than three separate linear layers, this model fuses them into one:

```python
# src/attention.py
self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)

# in forward():
qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
q, k, v = qkv.unbind(dim=2)
```

This is equivalent to three separate projections but uses one matrix multiply — more efficient
on GPUs where large matrix multiplies are cheap and sequential ops are expensive.

---

### 3.3.1 Causal Masking

**The problem:** Plain self-attention lets every token attend to every other token.
For language generation, that's cheating — token 5 can see token 10 during training
but not during inference.

**The fix:** Force attention weights to be zero for all future positions.

The causal mask is an upper-triangular matrix of `-inf` added to attention scores _before_
the softmax. After softmax, `-inf` → `0`, which means zero weight.

```
Mask for T=4:

     pos0  pos1  pos2  pos3
pos0 [  0   -inf  -inf  -inf ]
pos1 [  0    0   -inf  -inf ]
pos2 [  0    0    0   -inf ]
pos3 [  0    0    0    0   ]
```

In this model it is handled by PyTorch's built-in SDPA:

```python
# src/attention.py
y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=...,
    is_causal=True,   # ← applies the upper-triangular mask internally
)
```

`is_causal=True` also enables Flash Attention on supported hardware (NVIDIA Ampere+),
which computes attention in tiles without materializing the full `(T, T)` attention matrix.

**How GPT does it:** Identical mechanism. GPT-2 explicitly constructs the triangular mask tensor.
GPT-3 and later models rely on hardware-fused kernels like Flash Attention — same math.

---

### 3.4 Multi-Head Attention

**The intuition:** A single attention head can only learn one type of token relationship.
Multiple heads run in parallel, each free to attend to different patterns:
one head might track syntactic dependencies, another semantic similarity, another positional
proximity. The outputs are then concatenated and projected.

**The math:**

If `n_embd = 256` and `n_head = 4`, each head works in a `d_head = 64` dimensional subspace:

```
C = n_head × d_head
```

Each head has its own Q, K, V projections operating on 64 dimensions. After computing attention,
all heads' outputs are concatenated back to 256 dimensions and projected:

```python
# Reshape into heads: (B, T, C) → (B, n_head, T, d_head)
q = q.transpose(1, 2)  # (B, n_head, T, d_head)
k = k.transpose(1, 2)
v = v.transpose(1, 2)

# All n_head attention computations run in parallel as batched matmuls
y = F.scaled_dot_product_attention(q, k, v, ...)

# Reassemble: (B, n_head, T, d_head) → (B, T, C)
y = y.transpose(1, 2).contiguous().view(B, T, C)

# Output projection
y = self.proj(y)
```

The output projection is a learned linear layer that mixes information across all heads.

---

### 3.5 Transformer Block

**One block = attention + FFN, both with residual connections and LayerNorm.**

```python
# src/model.py
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # attention sub-layer
        x = x + self.ffn(self.ln2(x))    # feed-forward sub-layer
        return x
```

Three design choices to understand:

**1. Residual connections (`x = x + sublayer(x)`):**

The `+x` bypasses each sublayer, creating a direct gradient path from the loss back to the
earliest layers. Without residuals, gradients vanish in deep networks. With them, even a 100-layer
model trains effectively.

**2. Pre-LayerNorm (LN before the sublayer):**

The original transformer paper applied LN _after_ the sublayer (Post-LN). GPT-2 switched to
Pre-LN (LN _before_). Pre-LN is more stable to train because the residual stream remains
unnormalized — gradients flow without the normalization layer in the path.

**3. Feed-Forward Network (FFN):**

After attention mixes information _across_ tokens, the FFN processes each token _independently_
(same weights applied to each position). It expands the dimension 4×, applies GELU, then
projects back down:

```python
# src/model.py
class FeedForward(nn.Module):
    def __init__(self, n_embd, mult=4, dropout=0.0):
        self.net = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),   # expand: 256 → 1024
            nn.GELU(),                           # smooth non-linearity
            nn.Linear(mult * n_embd, n_embd),   # project back: 1024 → 256
            nn.Dropout(dropout),
        )
```

The 4× expansion is the standard (from "Attention is All You Need"). The FFN is often described
as "where the model stores factual knowledge" — the attention heads route information, the FFN
transforms it.

**Why GELU and not ReLU?** GELU is smoother near zero. Empirically it trains better for language
tasks. The difference is small — both work.

**Compare with GPT:** GPT-2 uses the identical Block design. The full GPT-2 (Small) model
stacks 12 such blocks. GPT-3 stacks 96.

---

### 3.6 Output Layer

After all transformer blocks, the representation passes through a final LayerNorm and then
a linear projection to vocabulary size:

```python
# src/model.py
self.ln_f = nn.LayerNorm(n_embd)
self.head = nn.Linear(n_embd, vocab_size, bias=False)

# in forward():
x = self.ln_f(x)
logits = self.head(x)   # (B, T, vocab_size)
```

**Logits:** The raw unnormalized scores. `logits[b, t, v]` = how much the model thinks
token `v` should follow the context `x[b, 0:t+1]`.

**During training** — cross-entropy loss:

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
    targets.view(-1),                   # (B*T,)
)
```

Cross-entropy = `-log(softmax(logits)[true_class])`. It is large when the model is confident
about the wrong token, small when it is confident about the right token. Minimizing it teaches
the model to assign high probability to the correct next token.

**During generation** — temperature + sampling:

```python
logits = logits[:, -1, :] / temperature   # take last position, scale
probs = torch.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
```

Temperature < 1.0 makes the distribution sharper (more deterministic).
Temperature > 1.0 makes it flatter (more random / creative).

---

## 4. Training Process

### The training loop step-by-step

```
for step in 1..N:
    1. Sample batch  (x, y) from ByteDataset         shape: (B, T)
    2. Forward pass  logits, loss = model(x, y)
    3. Zero gradients
    4. Backward pass loss.backward()
    5. Clip gradients  clip_grad_norm_(params, 1.0)
    6. Optimizer step  AdamW.step()
```

### Loss: cross-entropy

The model predicts a probability distribution over 256 tokens for every position.
Cross-entropy measures how many bits it takes to encode the true next token under that
distribution. A random model has `log2(256) = 8 bits = 2.08 nats` of loss. A perfect model has 0.

As training progresses, loss should drop from ~5.5 → ~1.5–2.0 on a small text file.

### Optimizer: AdamW

```python
opt = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),    # momentum terms
    weight_decay=0.1,     # L2 regularization on weights
)
```

AdamW = Adam + decoupled weight decay. Weight decay prevents weights from growing large,
acting as a regularizer. The β values `(0.9, 0.95)` are standard for transformer training.

### Gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

Before the optimizer step, all gradients are rescaled so their global L2 norm ≤ 1.0.
This prevents a single large-gradient step from destabilizing training. Essential when
training on short texts where individual batches can be highly non-representative.

### Checkpointing

Every `eval_interval` steps, the model is evaluated on a held-out validation split
(the last 10% of the file). If validation loss improves, a checkpoint is saved:

```python
torch.save({
    'model': model.state_dict(),
    'config': {...}    # saves arch params so the checkpoint is self-contained
}, 'runs/tiny-gpt/model_best.pt')
```

### Mixed-precision training (optional `--amp`)

On CUDA GPUs, `torch.cuda.amp.autocast` runs the forward pass in fp16 instead of fp32.
This halves memory bandwidth and speeds up matrix multiplies on Tensor Core hardware.
`GradScaler` scales up the loss before backward to prevent fp16 underflow, then unscales
before the optimizer step.

---

## 5. Key Differences vs GPT

### Scale comparison

| Aspect              | Tiny LLM (this) | GPT-2 Small  | GPT-3          |
| ------------------- | --------------- | ------------ | -------------- |
| Parameters          | ~3M             | 117M         | 175B           |
| Layers              | 4               | 12           | 96             |
| Attention heads     | 4               | 12           | 96             |
| Embedding dimension | 256             | 768          | 12,288         |
| Context window      | 256 tokens      | 1,024        | 2,048          |
| Vocabulary          | 256 (bytes)     | 50,257 (BPE) | 50,257 (BPE)   |
| Training data       | ~270KB          | 40GB         | ~300B tokens   |
| Training compute    | Minutes (1 CPU) | Weeks (TPUs) | Months (A100s) |

### Architectural differences

**Tokenization:**
This model uses a byte-level tokenizer (vocab = 256). GPT-2/GPT-3 use Byte Pair Encoding
which learns ~50K subword tokens. BPE is more efficient: a word like `running` might be
1 token instead of 7 bytes. But the transformer that processes those tokens is the same.

**No weight tying:**
GPT-2 ties the output projection (`head`) weights to the token embedding matrix (`tok_emb`):
`head.weight = tok_emb.weight`. This saves parameters and often improves perplexity.
This model keeps them separate for clarity.

**No LR schedule:**
Production GPT training uses a cosine learning rate schedule with a linear warmup.
This model uses a constant LR. For short training runs on small data, the difference is minor.

**Positional embeddings:**
This model uses learned absolute position embeddings. GPT-3 and LLaMA use Rotary Position
Embeddings (RoPE) which can generalize to longer sequences. The principle is the same.

**The core is identical:**
Every architectural choice that matters — Pre-LN transformer blocks, causal multi-head
self-attention, residual connections, 4× FFN expansion, no encoder — is shared with GPT.

---

## 6. Step-by-Step Data Flow

Tracing one forward pass with defaults: `B=4, T=256, n_embd=256, n_head=4, n_layer=4, vocab=256`.

```
Input: idx  (B=4, T=256)     ← integer token IDs

── Embeddings ──────────────────────────────────────────────────────────────
tok_emb(idx)                  (4, 256, 256)   ← look up each token
pos_emb(0..255)               (1, 256, 256)   ← look up each position
x = tok_emb + pos_emb         (4, 256, 256)   ← combined input to transformer

── Transformer Block 1 (of 4) ──────────────────────────────────────────────
  ln1(x)                      (4, 256, 256)   ← pre-norm
  qkv = Linear(256, 768)(x)   (4, 256, 768)   ← fused Q/K/V projection
  q, k, v = split(qkv)        each (4, 256, 64) per head, n_head=4
  q = q.T(1,2)                (4, 4, 256, 64) ← (B, n_head, T, d_head)
  k = k.T(1,2)                (4, 4, 256, 64)
  v = v.T(1,2)                (4, 4, 256, 64)
  scores = Q @ K.T / sqrt(64) (4, 4, 256, 256) ← attention matrix
  scores = causal_mask(scores) (4, 4, 256, 256) ← upper triangle = -inf
  weights = softmax(scores)   (4, 4, 256, 256) ← row sums to 1
  y = weights @ V             (4, 4, 256, 64)  ← attended values
  y = y.T(1,2).view(4,256,256)(4, 256, 256)   ← reassemble
  y = proj(y)                 (4, 256, 256)   ← output projection
  x = x + y                  (4, 256, 256)   ← residual

  ln2(x)                      (4, 256, 256)
  ffn: Linear → GELU → Linear (4, 256, 256) → (4, 256, 1024) → (4, 256, 256)
  x = x + ffn(x)              (4, 256, 256)   ← residual

── (same for Blocks 2, 3, 4) ───────────────────────────────────────────────

── Output ──────────────────────────────────────────────────────────────────
ln_f(x)                       (4, 256, 256)   ← final norm
head(x)                       (4, 256, 256)   ← Linear(256, 256)
logits                        (4, 256, 256)   ← (B, T, vocab_size)

── Loss ────────────────────────────────────────────────────────────────────
logits.view(-1, 256)          (1024, 256)     ← flatten B*T
targets.view(-1)              (1024,)
loss = cross_entropy(...)     scalar          ← single number to minimize
```

---

## 7. Code ↔ Concept Mapping

Every major concept in transformer theory maps directly to code here.

---

### Byte tokenization

**Concept:** Map text to integers. The simplest possible vocab: every byte is a token.

**Code** (`src/utils.py`):

```python
def encode(self, s: str) -> torch.Tensor:
    return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)
```

---

### Next-token prediction target construction

**Concept:** For language modeling, targets are inputs shifted by one position.
Each token predicts its successor.

**Code** (`src/dataset.py`):

```python
x = torch.stack([buf[i:i + self.block_size] for i in ix])
y = torch.stack([buf[i + 1:i + 1 + self.block_size] for i in ix])
```

---

### Token + positional embeddings

**Concept:** Convert discrete token IDs and positions into continuous vectors
that can be processed by linear algebra operations.

**Code** (`src/model.py`):

```python
pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
x = self.tok_emb(idx) + self.pos_emb(pos)
```

---

### Fused QKV projection

**Concept:** Project the input into query, key, and value spaces for attention.
Fused into one matmul for efficiency.

**Code** (`src/attention.py`):

```python
self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)

qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
q, k, v = qkv.unbind(dim=2)
```

---

### Scaled dot-product attention with causal mask

**Concept:** Compute attention scores = `softmax(QK^T / sqrt(d_k)) * V`.
Causal mask prevents attending to future tokens.

**Code** (`src/attention.py`):

```python
y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.dropout.p if self.training else 0.0,
    is_causal=True,    # ← this IS the causal mask
)
```

---

### Pre-LayerNorm residual block

**Concept:** Normalize before each sublayer, then add the residual.
This is the architecture that makes deep transformers trainable.

**Code** (`src/model.py`):

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))   # pre-LN + attention + residual
    x = x + self.ffn(self.ln2(x))    # pre-LN + FFN + residual
    return x
```

---

### 4× FFN expansion

**Concept:** Each token's representation is independently passed through a two-layer MLP
that expands and then contracts. This is where per-token processing (vs. cross-token
communication in attention) happens.

**Code** (`src/model.py`):

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # expand: 256 → 1024
    nn.GELU(),
    nn.Linear(4 * n_embd, n_embd),   # contract: 1024 → 256
    nn.Dropout(dropout),
)
```

---

### Cross-entropy loss

**Concept:** At every position, the model outputs a distribution over vocab.
Cross-entropy penalizes the log-probability assigned to the actual next token.

**Code** (`src/model.py`):

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
    targets.view(-1),                   # (B*T,) — true next tokens
)
```

---

### Top-k / nucleus sampling

**Concept:** During generation, restrict sampling to only likely tokens.
`top_k`: keep only the k highest-logit tokens.
`top_p` (nucleus): keep the smallest set of tokens summing to probability p.

**Code** (`src/utils.py`):

```python
if top_k is not None and top_k < V:
    topk_vals, _ = torch.topk(filtered, top_k, dim=-1)
    kth = topk_vals[:, -1].unsqueeze(-1)
    filtered[filtered < kth] = float('-inf')
```

---

### Weight initialization

**Concept:** GPT-2 convention: initialize all linear and embedding weights from N(0, 0.02).
This keeps initial activations in a good range for the chosen learning rate.

**Code** (`src/model.py`):

```python
def _init_weights(self, m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

---

## 8. How to Run

### Install dependencies

```bash
uv sync
```

### Training data

`data/tiny_hi.txt` and `data/tiny.txt` are already in the `data/` folder.
`train.py` and `eval_loss.py` default to `data/tiny_hi.txt` — no `--data` flag needed.

You can override with any UTF-8 text file:

```bash
uv run train.py --data data/tiny.txt
```

### Train (CPU, fast smoke run)

```bash
uv run train.py \
    --block_size 128 --n_layer 2 --n_head 2 --n_embd 128 \
    --steps 400 --eval_interval 100 --sample_every 100 \
    --batch_size 32
```

### Train (default config on tiny_hi.txt)

```bash
uv run train.py --steps 2000
```

### Generate text from a checkpoint

```bash
uv run sample.py \
    --ckpt runs/tiny-gpt/model_best.pt \
    --prompt "Once upon a time" \
    --tokens 300 \
    --temperature 0.8
```

### Evaluate validation loss

```bash
uv run eval_loss.py --ckpt runs/tiny-gpt/model_best.pt --iters 100
```

### Run tests

```bash
uv run pytest tests/
```

---

## 9. How to Extend This

### Make it bigger

Increase layers and embedding dimension. Parameters scale roughly as:

```
params ≈ 12 * n_layer * n_embd^2
```

| n_layer | n_embd | ~Params |
| ------- | ------ | ------- | ------------- |
| 4       | 256    | 3M      |
| 6       | 384    | 10M     |
| 12      | 768    | 85M     | ← GPT-2 Small |
| 24      | 1024   | 345M    | ← GPT-2 Large |

```bash
uv run train.py --data data/large.txt \
    --n_layer 6 --n_head 6 --n_embd 384 \
    --steps 10000 --amp
```

### Better tokenizer

Replace `ByteTokenizer` with a BPE tokenizer. The vocabulary increases from 256 to ~32K–50K,
which means shorter sequences and better sample efficiency:

```python
# Drop-in replacement in src/utils.py
import tiktoken
enc = tiktoken.get_encoding("gpt2")
# enc.encode(text) → list of ints
# enc.decode(ids)  → str
# enc.n_vocab      → 50257
```

You will need to update `vocab_size` in `GPTConfig` to match.

### Learning rate schedule

A cosine LR schedule with warmup significantly improves training on larger runs:

```python
def get_lr(step, warmup_steps=100, max_steps=2000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### Weight tying

GPT-2 ties the output head weights to the embedding matrix, which reduces parameters
and often improves perplexity:

```python
# in GPT.__init__() after creating tok_emb and head:
self.head.weight = self.tok_emb.weight
```

### Fine-tuning

After pretraining, load a checkpoint and continue training on a smaller, domain-specific
dataset with a lower learning rate:

```bash
# Load model_best.pt, train on domain data
python train.py --data data/domain.txt --steps 500 --lr 1e-4
# (You'd need to add --resume_ckpt to train.py's argparse for this)
```

### Rotary Position Embeddings (RoPE)

Replace learned `pos_emb` with RoPE to handle sequences longer than those seen during
training. RoPE is used by LLaMA, Mistral, and Falcon.

---

## Project Structure

```
tiny_gpt_llm/
│
├── src/
│   ├── __init__.py       # package exports
│   ├── attention.py      # CausalSelfAttention
│   ├── model.py          # FeedForward, Block, GPT
│   ├── dataset.py        # ByteDataset
│   ├── training.py       # estimate_loss, train()
│   └── utils.py          # ByteTokenizer, top_k_top_p_filtering
│
├── configs/
│   └── config.py         # GPTConfig dataclass with defaults
│
├── tests/
│   ├── test_tokenizer.py
│   └── test_dataset_shift.py
│
├── data/                 # place .txt training files here
│
├── train.py              # entry point: python train.py --data ...
├── sample.py             # generate text from a checkpoint
├── eval_loss.py          # evaluate validation loss
├── requirements.txt
└── README.md             ← you are here
```

---

## Results

### Training Metrics

**Short run** (`--block_size 128 --n_layer 2 --n_head 2 --n_embd 128 --steps 400 --batch_size 32`):

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 100  | 1.4410    | 1.4274   |
| 200  | 1.2650    | 1.2501   |
| 300  | 1.1748    | 1.1579   |
| 400  | 1.1158    | 1.1048   |

**Full run** (`--steps 2000`, default config: `n_layer=4, n_head=4, n_embd=256, block_size=256`):

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 200  | 1.1586    | 1.1421   |
| 400  | 0.9893    | 0.9830   |
| 600  | 0.9221    | 0.9218   |
| 800  | 0.8699    | 0.8774   |
| 1000 | 0.8188    | 0.8314   |
| 1200 | 0.7769    | 0.8005   |
| 1400 | 0.7440    | 0.7764   |
| 1600 | 0.7126    | 0.7512   |
| 1800 | 0.6878    | 0.7322   |
| 2000 | 0.6646    | 0.7233   |

**Observations:**
- Loss dropped steadily throughout with no spikes — AdamW + gradient clipping provided stable CPU training.
- Train/val gap starts widening around step 1200, indicating the onset of mild overfitting.
- By step 2000 the gap is ~0.06 nats, modest given the tiny 270KB dataset.
- No plateau was hit; loss was still declining at step 2000, suggesting more steps would help further.

### Sample Outputs

**Step 100 (short run):** mostly incoherent Devanagari character noise, no real words:
```
्संराखगसीाेभिआउ्ति  मसहकरगटेउ बिषमबान झषकेतू॥४नानत ुाईईअ्न प॥
```

**Step 400 (short run):** isolated valid words and partial verse fragments start appearing:
```
झावा। भावी बस न ग्यानु उर आवा॥
कह प्रभु जाहु जो बतूहं ु यभकि॥
```

**Step 1000 (full run):** recognizable Ramcharitmanas-style couplets with correct verse structure:
```
जानी। बोले अति सनेहमय बानी॥
उठहु राम भंजहु भवचापा। मेटहु तात जनक परितापा॥
```

**Step 2000 (full run):** coherent verse openings and valid doha/chaupai structure, degrades mid-stanza:
```
वहिं ते न परहिं भवकूपा॥
दो0-बिप्र धेनु सुर संत हित लीन्ह मनुज अवतार।
निज इच्छा निर्मित तनु माया गुर नमाथा॥
```

**Comments:**
- The model learned verse punctuation (`॥`, `दो0-`, `चौ०-`) and verse numbering — structural patterns far above raw character bigrams.
- Repetition is rare; the model is not degenerate or looping.
- Coherence within a line is good by step 1400+; cross-line semantic coherence breaks down quickly.
- Hindi compound verb forms and honorifics appear correctly within short spans.
- Prompting with English ("Once upon a time") yields Devanagari output immediately — the model knows only the byte distribution of its training corpus.

### Training Behavior

- Largest loss drop occurred in the first 400 steps (~1.4 → ~0.99); diminishing returns after.
- Validation loss closely tracked training loss until step ~1200, then lagged slightly — textbook generalization curve.
- 2000 steps on this dataset did not fully converge; more steps or more data would drive val loss further down.

---

## Analysis

### What the Model Learned

At 2000 steps the model has internalized:
- **UTF-8 byte structure** — never generates invalid Devanagari byte sequences.
- **Word-level patterns** — spaces, matras, and conjunct consonants appear in grammatically plausible positions.
- **Verse schema** — doha/chaupai formatting (`दो0-`, `चौ०-`, `॥` end markers, verse numbering) is reproduced far more often than chance.
- **Short phrase fluency** — individual half-lines are often indistinguishable from authentic Ramcharitmanas lines.

### Limitations

**Context window exhaustion:** Coherence breaks down after ~30–50 characters. With a 256-token context and byte-level encoding, 256 tokens is only ~64–85 Hindi characters — not enough for multi-sentence structure.

**No semantic grounding:** The model reproduces the surface form of verse but has no concept of meaning. `राम` and `रावण` are byte sequences; the model doesn't know they are characters with different story roles.

**Mild structural repetition:** Verse-ending punctuation (`॥`) is slightly over-represented because these bytes follow very high-frequency patterns in the corpus.

### Why These Limitations Exist

| Limitation | Root cause |
|---|---|
| Context coherence degrades fast | `n_layer=4, n_embd=256` limits representable contextual state |
| No semantic understanding | 3M parameters cannot compress a language's concept space |
| Short effective context | Byte tokenizer: 3–4 tokens per Hindi char → 256-token window ≈ ~70 chars |
| Mild overfitting after step 1200 | 270KB dataset; model can partially memorize it |
| Training not converged | Loss still declining at step 2000; more steps needed |

### Why Large GPT Models Perform Better

The same architecture with more resources eliminates all these issues:

- **More parameters** (GPT-2: 117M, GPT-3: 175B) → richer representations, can encode semantics and world knowledge.
- **Larger context** (GPT-3: 2048, GPT-4: 128K tokens) → paragraph-level coherence, not just line-level.
- **BPE tokenization** → `running` = 1 token, not 7 bytes; the same context window covers 5–10× more text.
- **Massive training data** (300B+ tokens) → each concept seen millions of times; surface patterns and semantics both learned.
- **Longer training + LR schedules** → loss driven far lower; outputs are indistinguishable from human writing at scale.

The architecture is not the bottleneck. Scale is.

---

## What I Implemented

**Dataset preparation:**
- Raw UTF-8 text (`data/tiny_hi.txt`, ~270KB Ramcharitmanas corpus)
- 90/10 train/val split loaded entirely into memory as a flat byte buffer (`src/dataset.py`)

**Tokenization:**
- `ByteTokenizer` — maps every UTF-8 byte to an integer in `[0, 255]`, fixed vocabulary of 256
- No learned merges, no special tokens — simplest tokenizer that handles any Unicode language (`src/utils.py`)

**Model architecture:**
- `GPTConfig` dataclass: `n_layer=4, n_head=4, n_embd=256, block_size=256, vocab_size=256`
- Token embeddings + learned absolute positional embeddings, summed into `(B, T, C)` input tensor
- `n_layer` stacked `Block` modules: Pre-LN → CausalSelfAttention → residual → Pre-LN → FeedForward → residual
- `CausalSelfAttention`: fused QKV projection, multi-head reshape, `F.scaled_dot_product_attention(is_causal=True)`, output projection
- `FeedForward`: `Linear(C → 4C) → GELU → Linear(4C → C) → Dropout`
- Final `LayerNorm` + `Linear(C → vocab_size)` output head (`src/model.py`, `src/attention.py`)

**Training loop** (`src/training.py`):
- AdamW optimizer (`lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1`)
- Gradient clipping at global norm 1.0
- Periodic validation loss estimation and best-checkpoint saving to `runs/tiny-gpt/model_best.pt`
- Optional AMP (`--amp`) with `GradScaler` for CUDA

**Sampling** (`sample.py`):
- Loads checkpoint, reconstructs model from saved config
- Autoregressive generation with temperature scaling and optional top-k/nucleus filtering

---

## Possible Improvements

### Model Improvements

- **More layers and wider embeddings** — parameters scale as `12 × n_layer × n_embd²`; going from `(4, 256)` to `(6, 384)` yields ~10M params. More layers give the model additional "reasoning steps" per forward pass. Worth trying before any other change.
- **Weight tying** — share `tok_emb.weight` and `head.weight` (GPT-2 default). Reduces parameters and empirically improves perplexity because the embedding and unembedding spaces are forced to align.
- **Rotary Position Embeddings (RoPE)** — learned absolute embeddings cannot extrapolate beyond `block_size`. RoPE encodes position as a relative rotation in the Q/K dot product, enabling generalization to longer sequences at inference time without retraining. Used by LLaMA, Mistral, Falcon.

### Data Improvements

- **Larger dataset** — the model is data-starved at 270KB. Even 5–10MB of Hindi text would significantly reduce overfitting and push val loss lower. The Mahabharata, Vedic texts, or a Hindi Wikipedia dump are natural sources.
- **Better preprocessing** — remove duplicate lines, normalize Unicode (NFC/NFKC), filter noise. Cleaner data means model capacity is spent on meaningful patterns.
- **Data augmentation** — shuffle verses, mix multiple source texts. Reduces the model's ability to memorize positional order of the corpus.

### Training Improvements

- **More steps** — training loss was still declining at step 2000; 5000–10000 steps would likely push val loss below 0.65.
- **Cosine LR schedule with warmup** — linear warmup for ~100 steps then cosine decay to `min_lr=3e-5`. The warmup prevents large early updates from destabilizing embeddings; cosine decay avoids over-shooting the optimum late in training.
- **Dropout regularization** — set `dropout=0.1` in attention and FFN to combat the overfitting visible after step 1200. Acts as stochastic regularization with no architectural overhead.
- **Gradient accumulation** — simulate larger effective batch sizes on CPU by accumulating gradients over N micro-batches before `optimizer.step()`. Larger batches reduce variance in gradient estimates and can improve convergence.

### Advanced Improvements

- **BPE tokenizer** — replace `ByteTokenizer` with tiktoken or SentencePiece. For Hindi text, BPE reduces average sequence length by ~3–4×: the same 256-token context window covers 200+ characters instead of ~70. This directly improves long-range coherence at zero extra compute cost.
- **Scaling laws** — Chinchilla (Hoffmann et al., 2022) shows the optimal token count is ~20× the parameter count. For a 3M param model: ~60M tokens. This dataset is ~270K tokens, roughly 220× below optimal. Scaling data matters more than scaling model size at this parameter count.
- **Fine-tuning / instruction tuning** — pretrain on a large corpus, then fine-tune on a small curated instruction dataset (lower LR, ~500 steps). The pretrained representations generalize; the fine-tune shapes generation toward a specific task (Q&A, translation, summarization).

---

## How This Compares to GPT

### Scale Difference

| Dimension        | This model                | GPT-2 Small    | GPT-3           |
|------------------|---------------------------|----------------|-----------------|
| Parameters       | ~3M                       | 117M           | 175B            |
| Layers           | 4                         | 12             | 96              |
| Embedding dim    | 256                       | 768            | 12,288          |
| Context window   | 256 tokens (~70 Hindi chars) | 1,024 tokens | 2,048 tokens  |
| Vocabulary       | 256 bytes                 | 50,257 (BPE)   | 50,257 (BPE)    |
| Training data    | ~270KB                    | 40GB WebText   | ~300B tokens    |
| Training compute | ~30 min, 1 CPU            | Weeks, TPU pods | Months, A100s  |

Parameters alone differ by 4 orders of magnitude between this model and GPT-3. That gap is the entire difference in capability.

### Architectural Similarities

The core is **identical** — if you understand this codebase, you understand the architecture of GPT-2, GPT-3, GPT-4, LLaMA, Mistral, and every other production decoder-only LLM:

- Decoder-only transformer (no encoder, no cross-attention)
- Pre-LayerNorm transformer blocks
- Causal multi-head self-attention with fused QKV projection
- 4× FFN expansion with GELU
- Residual connections at every sublayer
- Cross-entropy next-token prediction loss
- AdamW optimizer with gradient clipping

### Missing Components

| Component | GPT-2/3 | This model | Impact |
|---|---|---|---|
| BPE tokenizer | ✓ | ✗ (byte-level) | 3–4× longer sequences for same text |
| Weight tying | ✓ | ✗ | Slightly higher parameter count |
| Cosine LR schedule | ✓ | ✗ | Suboptimal convergence on long runs |
| Dropout | ✓ | optional (0.0 default) | More overfitting on small data |
| Scaled init by depth | ✓ (GPT-2) | ✗ | Minor stability benefit at 4 layers |
| RoPE / ALiBi | GPT-3+ / LLaMA | ✗ | Can't extrapolate beyond `block_size` |

None of these change the fundamental architecture — they are engineering refinements that matter at scale.

### Why GPT Performs Better

Three factors explain almost all of the gap:

1. **Data** — GPT-3 saw ~500,000× more text. Language understanding requires statistical evidence across millions of concepts, idioms, and domains. This model sees each verse a handful of times; GPT-3 sees every common phrase millions of times.

2. **Parameters** — More layers = more reasoning steps per forward pass. More embedding dimensions = richer internal representations. GPT-3's 96-layer model can compose abstract concepts; this 4-layer model matches surface patterns.

3. **Compute** — More gradient steps with a tuned LR schedule drives the loss far lower. GPT-3 training ran for weeks on thousands of A100s; this runs 30 minutes on a laptop CPU. The architecture's capacity is never fully utilized without sufficient updates.

The lesson: this is not a toy approximation of GPT. It is GPT, at the scale where you can watch it learn in real time on your laptop.

---
