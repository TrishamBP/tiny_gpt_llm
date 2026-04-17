"""training.py — Training loop and loss estimation.

Contains:
  estimate_loss() : run multiple forward passes on train/val without gradient
                    updates to get a stable loss estimate
  train()         : full training loop used by train.py entry-point
"""
from __future__ import annotations
import os
import time
import torch

from .model import GPT
from .dataset import ByteDataset
from .utils import ByteTokenizer


# ---------------------------------------------------------------------------
# Loss estimation (no gradient, for logging)
# ---------------------------------------------------------------------------

def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict[str, float]:
    """Estimate mean loss over `args.eval_iters` random batches for each split.

    Runs in eval mode (disables dropout, batch norm updates, etc.) and wraps
    everything in torch.no_grad() to avoid building a computation graph.

    Returns:
        {'train': float, 'val': float}
    """
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = ds.get_batch(split, args.batch_size, args.device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    """Train a GPT model and save checkpoints.

    Expects `args` to be an argparse.Namespace (or any object) with fields
    matching those defined in train.py's argument parser.

    Key steps per iteration:
      1. Sample a random batch from ByteDataset
      2. Forward pass → cross-entropy loss
      3. Zero gradients, scale+backprop (AMP optional), clip grads, step
      4. Periodically log loss, evaluate on val, save best checkpoint, sample
    """
    tok = ByteTokenizer()
    ds = ByteDataset(args.data, block_size=args.block_size)

    model = GPT(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(args.device)

    # Optional torch.compile (PyTorch 2.0+): reduces Python overhead
    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    # AdamW with weight decay — the standard GPT optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # GradScaler enables mixed-precision (fp16) training when --amp is set
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.type == 'cuda'))

    best_val = float('inf')
    t0 = time.time()
    model.train()

    for step in range(1, args.steps + 1):
        xb, yb = ds.get_batch('train', args.batch_size, args.device)

        with torch.cuda.amp.autocast(enabled=(args.amp and args.device.type == 'cuda')):
            _, loss = model(xb, yb)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        # Gradient clipping prevents exploding gradients
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(opt)
        scaler.update()

        # --- Logging ---
        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {(time.time() - t0):.1f}s")
            t0 = time.time()

        # --- Periodic validation + checkpointing ---
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt_path = f"{args.out_dir}/model_best.pt"
                os.makedirs(args.out_dir, exist_ok=True)
                torch.save({
                    'model': model.state_dict(),
                    'config': {
                        'vocab_size': tok.vocab_size,
                        'block_size': args.block_size,
                        'n_layer': args.n_layer,
                        'n_head': args.n_head,
                        'n_embd': args.n_embd,
                        'dropout': args.dropout,
                    }
                }, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")

        # --- Periodic text sample ---
        if args.sample_every > 0 and step % args.sample_every == 0:
            start = torch.randint(0, len(ds.train) - args.block_size - 1, (1,)).item()
            seed = ds.train[start:start + args.block_size].unsqueeze(0).to(args.device)
            out = model.generate(
                seed,
                max_new_tokens=args.sample_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            txt = tok.decode(out[0].cpu())
            print(
                "\n================ SAMPLE ================\n"
                + txt[-(args.block_size + args.sample_tokens):]
                + "\n=======================================\n"
            )

    # Final checkpoint
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'model': model.state_dict()}, f"{args.out_dir}/model_final.pt")
