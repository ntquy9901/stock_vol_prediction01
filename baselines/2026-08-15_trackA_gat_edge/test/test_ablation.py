import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_ablation as ra  # noqa: E402


@pytest.mark.smoke
def test_run_horizon_leave_one_out_rungs(tmp_path, monkeypatch):
    monkeypatch.setattr(ra, "ROOT", tmp_path)

    def _fake_basis(stamp):
        N, seq = 4, 22

        def snap(split):
            return {"price": torch.randn(N, seq, 5), "news": torch.randn(N, seq, 146),
                    "news_mask": torch.ones(N, seq), "ticker_ids": torch.arange(N),
                    "adjacency": torch.ones(N, N), "target": torch.rand(N) + 0.1,
                    "target_raw": (torch.rand(N) + 0.1).tolist(),
                    "presence_mask": torch.ones(N), "split": split}
        snaps = [snap("train"), snap("train"), snap("val"), snap("val"), snap("test"), snap("test")]
        six = {k: 0.5 for k in ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")}
        return {"snaps": snaps, "num_tickers": N, "scaler_mean": torch.zeros(N),
                "scaler_std": torch.full((N,), 1e-4), "price_dim": 5, "news_dim": 146,
                "har": {"val": six, "test": six, "floor_hit_fraction": 0.0}}
    monkeypatch.setattr(ra, "build_trackA_basis", _fake_basis)
    ladder = ra.run_horizon(5, seed=0, epochs=1, ts="T", device=torch.device("cpu"))
    assert set(ladder["rungs"]) == {"HAR", "FULL", "minus_graph", "minus_gate", "minus_news"}
    assert set(ladder["leave_one_out_effects"]) == {"graph", "gate", "news"}
    assert (tmp_path / "results").exists()
