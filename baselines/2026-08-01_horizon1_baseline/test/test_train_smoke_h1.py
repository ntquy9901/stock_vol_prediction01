"""Smoke tests: one real train_epoch/validate step at forecast_horizon=1 for BOTH models, plus
the REAL-DATA, FULL-UNIVERSE window-count check (mirrors the h10/h22 siblings' equivalent test,
kept for consistency even though this horizon needs only a 23-day minimum per split -- the
LOWEST of all 4 horizons tried today, so no window-count issue is expected).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1] / "code"
_GATE_SIBLING_CODE = _ROOT / "baselines" / "2026-07-26_per_ticker_news_gate_baseline" / "code"
_DUAL_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_GATE_SIBLING_CODE), str(_DUAL_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from dataset_dual_news import MultiStockDatasetWithDualNews, HAR_COLS  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from src.lstm_gat_hybrid.model_parallel import ParallelLSTMGNN  # noqa: E402
from model_per_ticker_gate import PerTickerGatedNewsBaseline  # noqa: E402 (sibling, unchanged)

from train_har_only_reference_h1 import train_epoch as train_epoch_har_only  # noqa: E402
from train_per_ticker_gate_h1 import train_epoch as train_epoch_gate  # noqa: E402

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB", "CCC"]
N_DAYS = 90
FORECAST_HORIZON = 1


def _make_har_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_variance": rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[0]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[1]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[2]: rng.uniform(0.001, 0.05, N_DAYS),
    })


def _build_ds():
    stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithDualNews(
        stock_data, stock_data, STOCKS,
        seq_length=22, forecast_horizon=FORECAST_HORIZON, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None,
    )
    assert len(ds) > 0, "90-day synthetic series must still yield >=1 window at horizon=1"
    return ds, config


class TestHarOnlyH1Smoke:
    def test_train_epoch_end_to_end(self):
        ds, config = _build_ds()
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = ParallelLSTMGNN(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        loss = train_epoch_har_only(model, loader, criterion, optimizer, device="cpu")
        assert np.isfinite(loss)


class TestPerTickerGateH1Smoke:
    def test_train_epoch_end_to_end(self):
        ds, config = _build_ds()
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = PerTickerGatedNewsBaseline(config, n_feat=146, num_stocks=len(STOCKS), d_news=8,
                                           dropout=0.0)
        optimizer = torch.optim.Adam([
            {"params": [model.gate_logits], "lr": 0.05},
            {"params": [p for n, p in model.named_parameters() if n != "gate_logits"], "lr": 1e-3},
        ])
        criterion = torch.nn.MSELoss()
        loss = train_epoch_gate(model, loader, criterion, optimizer, device="cpu")
        assert np.isfinite(loss)
        gate_after = model.gate_values().numpy()
        assert gate_after.shape == (len(STOCKS),)
        assert np.all((gate_after > 0) & (gate_after < 1))


class TestRealDataFullUniverseWindowCount:
    """Lowest-risk horizon of the 4 tried today (design.md §3): only a 23-day minimum per split
    (vs 44 for horizon22, 32 for horizon10, 27 for horizon5). Checks the ACTUAL price data for
    every ticker `create_dual_news_dataloaders` would use, kept for consistency with the other
    horizon baselines' equivalent test even though no failure is expected here."""

    DATA_DIR = _ROOT / "data" / "processed"

    @pytest.mark.skipif(not DATA_DIR.exists(), reason="real price data directory not present")
    def test_every_common_stock_has_positive_windows_at_horizon_1(self):
        sys.path.insert(0, str(_ROOT))
        from src.lstm_gat_hybrid.dataset_with_graph_method import (
            _load_raw_stock_data, _split_raw_data_by_date, _generate_har_for_split,
        )

        seq_length = 22
        min_required_days = seq_length + FORECAST_HORIZON  # 23

        stock_data_raw = _load_raw_stock_data(data_dir=str(self.DATA_DIR))
        train_raw, val_raw, test_raw, _, _, _ = _split_raw_data_by_date(stock_data_raw)
        train_har = _generate_har_for_split(train_raw, 'train')
        val_har = _generate_har_for_split(val_raw, 'val')
        test_har = _generate_har_for_split(test_raw, 'test')
        common_stocks = sorted(set(train_har.keys()) & set(val_har.keys()) & set(test_har.keys()))
        assert len(common_stocks) > 0, "no common stocks found across all 3 splits"

        failures = []
        for split_name, split_har in (("train", train_har), ("val", val_har), ("test", test_har)):
            min_length = min(len(split_har[s]) for s in common_stocks)
            n_windows = min_length - seq_length - FORECAST_HORIZON
            if n_windows <= 0:
                failures.append(f"{split_name}: min_length={min_length} -> {n_windows} windows "
                                 f"(need > 0, min_required_days={min_required_days})")

        assert not failures, (
            "At least one split would yield 0 usable windows at forecast_horizon=1:\n"
            + "\n".join(failures)
        )
