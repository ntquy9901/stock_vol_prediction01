"""FU3a: per-layer MAD over test snapshots (over-smoothing by GAT depth)."""
import sys
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import mad_report  # noqa: E402
from model import VolatilityModel  # noqa: E402


def _snap(n=4, seq=4, pf=5, nf=6):
    return {
        "price": torch.randn(n, seq, pf), "adjacency": torch.eye(n),
        "presence_mask": torch.ones(n),
    }


def test_layer_mads_returns_one_value_per_gat_layer():
    torch.manual_seed(0)
    m2 = VolatilityModel(price_dim=5, news_dim=6, num_tickers=4, hidden=8, heads=2, gat_layers=2)
    snaps = [_snap() for _ in range(3)]
    out = mad_report.layer_mads(m2, snaps, torch.device("cpu"))
    assert set(out) == {1, 2}                          # gat1 + gat2 outputs
    for v in out.values():
        assert 0.0 <= v <= 2.0                         # cosine distance range


def test_layer_mads_one_hop_has_single_layer():
    m1 = VolatilityModel(price_dim=5, news_dim=6, num_tickers=4, hidden=8, heads=2, gat_layers=1)
    out = mad_report.layer_mads(m1, [_snap()], torch.device("cpu"))
    assert set(out) == {1}


def test_layer_mads_respects_presence():
    m = VolatilityModel(price_dim=5, news_dim=6, num_tickers=4, hidden=8, heads=2, gat_layers=2)
    snap = _snap()
    snap["presence_mask"] = torch.tensor([1.0, 1.0, 0.0, 0.0])   # 2 present -> MAD defined, finite
    out = mad_report.layer_mads(m, [snap], torch.device("cpu"))
    assert all(v == v for v in out.values())            # not NaN
