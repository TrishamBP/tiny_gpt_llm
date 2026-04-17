"""sample.py — Generate text from a trained checkpoint.

Usage:
    python sample.py --ckpt runs/tiny-gpt/model_best.pt --prompt "Once upon a time"
    python sample.py --ckpt runs/tiny-gpt/model_best.pt --tokens 400 --temperature 0.8 --top_k 40
"""
from __future__ import annotations
import argparse
import torch

from src.model import GPT
from src.utils import ByteTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',        type=str,   required=True)
    p.add_argument('--prompt',      type=str,   default='')
    p.add_argument('--tokens',      type=int,   default=200)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k',       type=int,   default=50)
    p.add_argument('--top_p',       type=float, default=None)
    p.add_argument('--cpu',         action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()

    # Encode prompt; fall back to newline byte if empty
    prompt_ids = tok.encode(args.prompt).unsqueeze(0).to(device)
    if prompt_ids.numel() == 0:
        prompt_ids = torch.tensor([[10]], dtype=torch.long, device=device)

    # Load checkpoint — config may or may not be present
    ckpt = torch.load(args.ckpt, map_location=device)
    config = ckpt.get('config', None)

    if config is None:
        model = GPT(tok.vocab_size, block_size=256).to(device)
    else:
        model = GPT(**config).to(device)

    model.load_state_dict(ckpt['model'])

    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    print(tok.decode(out[0].cpu()))


if __name__ == '__main__':
    main()
