"""Smoke test: boot the full EDA-GNN ladder (E0..E3 + controls) on a tiny real-data subset."""

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import eda_ladder

_ROOT = Path(__file__).resolve().parents[3]
_TICKERS = ("ACB", "BID", "CTG", "FPT")  # four VN30 tickers that have OHLCV volume
_ROWS = 360


def _stage_subset(tmp_path: Path) -> tuple[Path, Path]:
    processed = tmp_path / "processed"
    prices = tmp_path / "prices"
    processed.mkdir()
    prices.mkdir()
    for ticker in _TICKERS:
        proc = pd.read_csv(_ROOT / "data" / "processed" / f"{ticker}_processed.csv").iloc[:_ROWS]
        proc.to_csv(processed / f"{ticker}_processed.csv", index=False)
        ohlcv = pd.read_csv(_ROOT / "data" / "raw" / "prices" / f"{ticker}_ohlcv.csv")
        ohlcv = ohlcv[ohlcv["date"].isin(set(proc["date"]))]
        ohlcv.to_csv(prices / f"{ticker}_ohlcv.csv", index=False)
    return processed, prices


@pytest.mark.smoke
def test_ladder_boots_on_subset(tmp_path, monkeypatch):
    processed, prices = _stage_subset(tmp_path)
    monkeypatch.setattr(eda_ladder, "_PROCESSED", processed)
    monkeypatch.setattr(eda_ladder, "_PRICE_DIR", prices)
    monkeypatch.setattr(eda_ladder, "EPOCHS", 1)
    monkeypatch.setattr(eda_ladder, "GRAPH_BATCH", 16)

    stamp = tmp_path / "stamp"
    basis = eda_ladder.build_basis(stamp)
    out = tmp_path / "run" / "h5"
    eda_ladder.run_seed(basis, out, seed=42, device=torch.device("cpu"), stamp=stamp)

    ladder = json.loads((out / "ladder_metrics.json").read_text(encoding="utf-8"))
    rungs = ladder["rungs"]
    for name in ("E0", "E1", "E2", "E3", "E3off", "G1corr"):
        assert name in rungs
        for split in ("validation_metrics", "test_metrics"):
            metrics = rungs[name][split]
            for key in ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy"):
                assert key in metrics and metrics[key] == metrics[key]  # finite (not NaN)
    # per-observation test dumps exist for the DM step
    for name in ("E0", "E1", "E2", "E3", "E3_off", "G1corr"):
        assert (out / name / "predictions_test.json").exists()
