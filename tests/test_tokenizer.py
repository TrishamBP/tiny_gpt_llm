import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.utils import ByteTokenizer


def test_roundtrip():
    tok = ByteTokenizer()
    s = "Hello, ByteTok! äö"
    ids = tok.encode(s)
    assert ids.dtype == torch.long
    s2 = tok.decode(ids)
    assert len(s2) > 0


def test_vocab_size():
    tok = ByteTokenizer()
    assert tok.vocab_size == 256
