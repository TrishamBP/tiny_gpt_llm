"""train.py — Entry point for training Tiny GPT.

Usage:
    # Default: trains on data/tiny_hi.txt
    python train.py

    # Smoke run (fast, CPU-friendly)
    python train.py --steps 300 --sample_every 100

    # Override data file
    python train.py --data data/tiny.txt --steps 300

    # Full config control
    python train.py --data data/tiny_hi.txt \\
        --block_size 256 --n_layer 4 --n_head 4 --n_embd 256 \\
        --steps 2000 --lr 3e-4 --batch_size 32
"""
from __future__ import annotations
import argparse
import torch

from src.training import train


def parse_args():
    p = argparse.ArgumentParser(description='Train a tiny GPT language model')

    # Data
    p.add_argument('--data',          type=str,   default='data/tiny_hi.txt',   help='Path to training text file')
    p.add_argument('--out_dir',       type=str,   default='runs/tiny-gpt')

    # Architecture
    p.add_argument('--block_size',    type=int,   default=256)
    p.add_argument('--n_layer',       type=int,   default=4)
    p.add_argument('--n_head',        type=int,   default=4)
    p.add_argument('--n_embd',        type=int,   default=256)
    p.add_argument('--dropout',       type=float, default=0.0)

    # Optimization
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--steps',         type=int,   default=2000)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--weight_decay',  type=float, default=0.1)
    p.add_argument('--grad_clip',     type=float, default=1.0)

    # Evaluation
    p.add_argument('--eval_interval', type=int,   default=200)
    p.add_argument('--eval_iters',    type=int,   default=50)

    # Sampling during training
    p.add_argument('--sample_every',  type=int,   default=200)
    p.add_argument('--sample_tokens', type=int,   default=256)
    p.add_argument('--temperature',   type=float, default=1.0)
    p.add_argument('--top_k',         type=int,   default=50)
    p.add_argument('--top_p',         type=float, default=None)

    # Hardware
    p.add_argument('--cpu',     action='store_true', help='Force CPU even if CUDA available')
    p.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    p.add_argument('--amp',     action='store_true', help='Mixed-precision training (fp16)')

    args = p.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    return args


if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {args.device}")
    train(args)
