import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_volatility as rt  # noqa: E402


@pytest.mark.smoke
def test_run_seed_smoke(tmp_path, monkeypatch):
    def _fake_basis(stamp):
        N, seq = 4, 22

        def snap(split, date):
            # evaluate_records (reused unmodified from the pilot) requires >= 2 target dates
            # per ticker per split to compute directional accuracy, so val/test need 2 snaps each.
            return {"price": torch.randn(N, seq, 5), "news": torch.randn(N, seq, 146),
                     "news_mask": torch.ones(N, seq), "ticker_ids": torch.arange(N),
                     "adjacency": torch.ones(N, N), "target": torch.rand(N) + 0.1,
                     "target_raw": (torch.rand(N) + 0.1).tolist(),
                     "presence_mask": torch.ones(N), "split": split, "target_date": date}

        snaps = [snap("train", "2024-01-01"), snap("train", "2024-01-02"),
                 snap("val", "2024-02-01"), snap("val", "2024-02-02"),
                 snap("test", "2024-03-01"), snap("test", "2024-03-02")]
        return {"snaps": snaps, "num_tickers": N, "scaler_mean": torch.zeros(N),
                "scaler_std": torch.full((N,), 1e-4), "price_dim": 5, "news_dim": 146,
                "har": {"val": {k: 0.5 for k in ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")},
                        "test": {k: 0.5 for k in ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")}}}

    monkeypatch.setattr(rt, "build_volatility_basis", _fake_basis)
    out = rt.run_seed(seed=0, epochs=1, ts="T", out_base=tmp_path, device=torch.device("cpu"))
    assert set(out["rungs"]) >= {"HAR", "NODE", "GNN"}
    assert (tmp_path / "ckpt.pt").exists()
    for r in ("HAR", "NODE", "GNN"):
        assert "qlike" in out["rungs"][r]["test_metrics"]
