import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import train_resume as tr  # noqa: E402
from model import VolatilityModel  # noqa: E402


def _snap(n=3, seq=4, pf=5, nf=6, present=None):
    return {
        "price": torch.randn(n, seq, pf),
        "news": torch.randn(n, seq, nf),
        "news_mask": torch.ones(n, seq),
        "ticker_ids": torch.arange(n, dtype=torch.long),
        "adjacency": torch.eye(n),
        "target": torch.randn(n),
        "presence_mask": torch.ones(n) if present is None else present,
        "split": "train",
    }


def test_collate_stacks_batch_dim():
    snaps = [_snap(), _snap(), _snap()]
    b = tr._collate(snaps, torch.device("cpu"))
    assert b["price"].shape == (3, 3, 4, 5)
    assert b["news"].shape == (3, 3, 4, 6)
    assert b["news_mask"].shape == (3, 3, 4)
    assert b["ticker_ids"].shape == (3, 3)
    assert b["adjacency"].shape == (3, 3, 3)
    assert b["target"].shape == (3, 3)
    assert b["presence_mask"].shape == (3, 3)


def test_masked_mse_ignores_absent_nodes():
    pred = torch.zeros(1, 3)
    target = torch.tensor([[0.0, 0.0, 100.0]])   # node 2 has huge error but is absent
    presence = torch.tensor([[1.0, 1.0, 0.0]])
    loss = tr._masked_mse(pred, target, presence)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_masked_mse_matches_plain_mean_when_all_present():
    pred = torch.tensor([[1.0, 2.0]])
    target = torch.tensor([[0.0, 0.0]])
    presence = torch.ones(1, 2)
    loss = tr._masked_mse(pred, target, presence)
    assert torch.isclose(loss, torch.tensor(2.5), atol=1e-6)   # (1+4)/2


def test_batched_train_runs_and_checkpoints(tmp_path):
    torch.manual_seed(0)
    model = VolatilityModel(price_dim=5, news_dim=6, num_tickers=3, hidden=8, heads=2)
    model.configure_positivity(torch.zeros(3), torch.ones(3))
    snaps = [_snap() for _ in range(4)]
    ckpt = tmp_path / "m.pt"
    out = tr.train_with_resume(model, snaps, snaps, ckpt, epochs=1, device=torch.device("cpu"),
                               seed=0, batch_size=2)
    assert ckpt.exists()
    assert out["epoch"] == 1
    import math
    assert math.isfinite(out["best_val"])
