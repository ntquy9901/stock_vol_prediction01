"""Smoke tests: one real train_epoch/validate step at forecast_horizon=10 for BOTH models
(ParallelLSTMGNN HAR-only, PerTickerGatedNewsBaseline), on synthetic data (mirrors this session's
established `test_train_smoke.py` pattern). Confirms the whole path -- dataset with the new
horizon, model forward/backward, gate debug helpers -- works end to end before spending time on a
real 10-epoch training run.
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

from train_har_only_reference_h10 import train_epoch as train_epoch_har_only  # noqa: E402
from train_per_ticker_gate_h10 import (  # noqa: E402
    train_epoch as train_epoch_gate, print_gate_table, plot_gate_evolution,
)

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB", "CCC"]
N_DAYS = 60
FORECAST_HORIZON = 10


def _make_har_df(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
    return pd.DataFrame({
        "date": dates.astype(str),
        "parkinson_volatility": rng.uniform(0.001, 0.05, N_DAYS),
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
        seq_length=10, forecast_horizon=FORECAST_HORIZON, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None,
    )
    assert len(ds) > 0, "synthetic 60-day series must still yield >=1 window at horizon=10"
    return ds, config


class TestHarOnlyH10Smoke:
    def test_train_epoch_end_to_end(self):
        ds, config = _build_ds()
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        model = ParallelLSTMGNN(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        loss = train_epoch_har_only(model, loader, criterion, optimizer, device="cpu")
        assert np.isfinite(loss)


class TestPerTickerGateH10Smoke:
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

    def test_gate_debug_helpers_still_work(self, capsys, tmp_path):
        print_gate_table(STOCKS, np.array([0.9, 0.1, 0.5]), None, epoch=1)
        assert "AAA" in capsys.readouterr().out

        gate_history = {"1": {sn: 0.5 for sn in STOCKS}, "5": {sn: 0.6 for sn in STOCKS}}
        out_path = tmp_path / "gate_evolution.png"
        plot_gate_evolution(gate_history, STOCKS, out_path)
        assert out_path.exists() and out_path.stat().st_size > 0


class TestRealDataSliceWindowCount:
    """Real-data-sample smoke (CLAUDE.md Testing quality rules): confirms window count at
    horizon=10 shrinks by exactly 5 relative to horizon=5 on an ACTUAL price CSV slice, not just
    synthetic data (design.md §7 risk: 'windows shrink slightly, should stay usable')."""

    REAL_PRICE_FILE = _ROOT / "data" / "processed" / "ACB_processed.csv"

    @pytest.mark.skipif(not REAL_PRICE_FILE.exists(), reason="real price CSV not present")
    def test_window_count_shrinks_by_5_on_real_slice(self):
        sys.path.insert(0, str(_ROOT))
        from src.common.feature_engineering import create_har_features

        raw = pd.read_csv(self.REAL_PRICE_FILE).head(120)
        har = create_har_features(raw["parkinson_volatility"])
        combined = pd.concat([raw[["date", "parkinson_volatility"]], har], axis=1).dropna()
        combined = combined.rename(columns={
            "har_daily_vol": HAR_COLS[0], "har_weekly_vol": HAR_COLS[1], "har_monthly_vol": HAR_COLS[2],
        }).reset_index(drop=True)
        assert len(combined) > 40

        stock_data = {"ACB": combined, "ACB2": combined}
        config = LSTMGATConfig()
        config.num_features_per_stock = 3

        ds5 = MultiStockDatasetWithDualNews(
            stock_data, stock_data, ["ACB", "ACB2"], seq_length=10, forecast_horizon=5,
            graph_method="correlation", normalize=False, config=config, news_panel_path=None)
        ds10 = MultiStockDatasetWithDualNews(
            stock_data, stock_data, ["ACB", "ACB2"], seq_length=10, forecast_horizon=10,
            graph_method="correlation", normalize=False, config=config, news_panel_path=None)

        assert len(ds10) == len(ds5) - 5
        assert len(ds10) > 0
