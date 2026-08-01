"""Smoke tests: one real train_epoch/validate step with PerTickerGatedNewsBaseline (UNCHANGED,
imported read-only from the 2026-07-26 sibling) on the calendar-augmented dataset, plus the
debug-output helpers (mirrors `2026-07-26_per_ticker_news_gate_baseline/test/test_train_smoke.py`
1:1, since this baseline's train script duplicates that logic unchanged -- design.md §6).
"""
import sys
import json
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

from dataset_calendar_news import MultiStockDatasetWithCalendarNews, HAR_COLS, N_CALENDAR_FEAT  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from model_per_ticker_gate import PerTickerGatedNewsBaseline  # noqa: E402 (sibling, unchanged)
from train_calendar_news_gate import (  # noqa: E402
    print_gate_table, plot_gate_evolution, train_epoch, load_resume_state,
)

pytestmark = pytest.mark.smoke

STOCKS = ["AAA", "BBB", "CCC"]
N_DAYS = 60


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


def test_print_gate_table_runs_without_crash(capsys):
    gate = np.array([0.9, 0.1, 0.5])
    prev = np.array([0.8, 0.2, 0.5])
    print_gate_table(STOCKS, gate, prev, epoch=3)
    out = capsys.readouterr().out
    assert "AAA" in out and "BBB" in out and "CCC" in out
    assert "epoch 3" in out


def test_plot_gate_evolution_writes_file(tmp_path):
    gate_history = {
        "1": {"AAA": 0.5, "BBB": 0.5, "CCC": 0.5},
        "5": {"AAA": 0.7, "BBB": 0.3, "CCC": 0.5},
    }
    out_path = tmp_path / "gate_evolution.png"
    plot_gate_evolution(gate_history, STOCKS, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_load_resume_state_no_resume_returns_empty():
    gate_history, train_losses, val_losses, start_epoch = load_resume_state(None)
    assert gate_history == {} and train_losses == [] and val_losses == []
    assert start_epoch == 0


def test_load_resume_state_missing_gate_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resume_state(tmp_path)


def test_train_epoch_smoke_end_to_end():
    """One real forward+backward+optimizer.step() with the UNCHANGED PerTickerGatedNewsBaseline,
    fed calendar-augmented x_news (146 dummy dual-group dims + N_CALENDAR_FEAT real calendar
    dims, since news_panel_path=None here)."""
    stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithCalendarNews(
        stock_data, stock_data, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None,
    )
    assert ds._n_feat == 146 + N_CALENDAR_FEAT
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = PerTickerGatedNewsBaseline(config, n_feat=ds._n_feat, num_stocks=len(STOCKS), d_news=8,
                                       dropout=0.0)
    optimizer = torch.optim.Adam([
        {"params": [model.gate_logits], "lr": 0.05},
        {"params": [p for n, p in model.named_parameters() if n != "gate_logits"], "lr": 1e-3},
    ])
    criterion = torch.nn.MSELoss()

    loss = train_epoch(model, loader, criterion, optimizer, device="cpu")
    assert np.isfinite(loss)

    gate_after = model.gate_values().numpy()
    assert gate_after.shape == (len(STOCKS),)
    assert np.all((gate_after > 0) & (gate_after < 1))
    print(f"train_epoch smoke OK: loss={loss:.6f}, gate={gate_after}")
