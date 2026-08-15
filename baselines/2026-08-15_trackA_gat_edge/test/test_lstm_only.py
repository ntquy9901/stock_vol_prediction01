import sys
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import run_lstm_only as rlo  # noqa: E402


@pytest.mark.smoke
def test_run_horizon_lstm_only(tmp_path, monkeypatch):
    monkeypatch.setattr(rlo, "ROOT", tmp_path)

    def _fake_basis(stamp):
        N, seq = 4, 22

        def snap(split):
            return {"price": torch.randn(N, seq, 5), "news": torch.randn(N, seq, 146),
                    "news_mask": torch.ones(N, seq), "ticker_ids": torch.arange(N),
                    "adjacency": torch.ones(N, N), "target": torch.rand(N) + 0.1,
                    "target_raw": (torch.rand(N) + 0.1).tolist(),
                    "presence_mask": torch.ones(N), "split": split}
        snaps = [snap("train"), snap("train"), snap("val"), snap("val"), snap("test"), snap("test")]
        return {"snaps": snaps, "num_tickers": N, "scaler_mean": torch.zeros(N),
                "scaler_std": torch.full((N,), 1e-4), "price_dim": 5, "news_dim": 146}
    monkeypatch.setattr(rlo, "build_trackA_basis", _fake_basis)
    metrics = rlo.run_horizon(5, seed=0, epochs=1, ts="T", device=torch.device("cpu"))
    assert set(metrics) == {"validation_metrics", "test_metrics", "floor_hit_fraction"}
    assert (tmp_path / "results").exists()
    # the LSTM-only checkpoint and its own dump dir were written under the shared result dir
    out = tmp_path / "results" / "trackA_ablation_h5_seed0_T"
    assert (out / "lstm_only.pt").exists()
    assert (out / "lstm_only_metrics.json").exists()
