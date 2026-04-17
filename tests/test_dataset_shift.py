import torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.dataset import ByteDataset


def test_shift_alignment(tmp_path):
    p = tmp_path / 'toy.txt'
    p.write_text('abcdefg')
    ds = ByteDataset(str(p), block_size=3, split=1.0)
    x, y = ds.get_batch('train', 2, device=torch.device('cpu'))
    # y must be x shifted right by one position (next-token prediction)
    assert (y[:, :-1] == x[:, 1:]).all()
