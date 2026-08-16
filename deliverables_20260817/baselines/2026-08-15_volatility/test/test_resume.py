import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from model import VolatilityModel          # noqa: E402
from train_resume import load_checkpoint, train_with_resume  # noqa: E402


def _snaps(n_snap=3, N=4, seq=22):
    torch.manual_seed(0)
    snaps = []
    for _ in range(n_snap):
        snaps.append({
            "price": torch.randn(N, seq, 5), "news": torch.randn(N, seq, 146),
            "news_mask": torch.ones(N, seq), "ticker_ids": torch.arange(N),
            "adjacency": torch.eye(N), "target": torch.rand(N) + 0.1,
        })
    return snaps


def test_train_then_resume_advances_epoch(tmp_path):
    m = VolatilityModel(5, 146, 4)
    ckpt = tmp_path / "m.pt"
    train_with_resume(m, _snaps(), _snaps(), ckpt, epochs=2, device=torch.device("cpu"), seed=0)
    assert load_checkpoint(ckpt)["epoch"] == 2
    m2 = VolatilityModel(5, 146, 4)
    train_with_resume(m2, _snaps(), _snaps(), ckpt, epochs=1, device=torch.device("cpu"),
                      seed=0, resume=True)
    assert load_checkpoint(ckpt)["epoch"] == 3          # resumed, not restarted
