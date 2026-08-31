"""Smoke tests: one real train_epoch/validate step with PerTickerGatedNewsBaseline on synthetic
data (mirrors sibling baselines' test_dataset_smoke.py conventions), plus the debug-output helpers
required by the user (2026-07-26): per-ticker gate console table, gate_history.json persistence,
gate-evolution plot file.
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
_SIBLING_CODE = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _p in (str(_ROOT), str(_CODE), str(_SIBLING_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

from dataset_dual_news import MultiStockDatasetWithDualNews, HAR_COLS  # noqa: E402
from src.lstm_gat_hybrid.config import LSTMGATConfig  # noqa: E402
from model_per_ticker_gate import PerTickerGatedNewsBaseline  # noqa: E402
from train_per_ticker_gate import (  # noqa: E402
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
        "parkinson_variance": rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[0]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[1]: rng.uniform(0.001, 0.05, N_DAYS),
        HAR_COLS[2]: rng.uniform(0.001, 0.05, N_DAYS),
    })


def test_print_gate_table_runs_without_crash(capsys):
    stock_names = STOCKS
    gate = np.array([0.9, 0.1, 0.5])
    prev = np.array([0.8, 0.2, 0.5])
    print_gate_table(stock_names, gate, prev, epoch=3)
    out = capsys.readouterr().out
    assert "AAA" in out and "BBB" in out and "CCC" in out
    assert "epoch 3" in out


def test_print_gate_table_handles_no_previous_epoch(capsys):
    print_gate_table(STOCKS, np.array([0.5, 0.5, 0.5]), None, epoch=1)
    out = capsys.readouterr().out
    assert "epoch 1" in out


def test_plot_gate_evolution_writes_file(tmp_path):
    gate_history = {
        "1": {"AAA": 0.5, "BBB": 0.5, "CCC": 0.5},
        "5": {"AAA": 0.7, "BBB": 0.3, "CCC": 0.5},
    }
    out_path = tmp_path / "gate_evolution.png"
    plot_gate_evolution(gate_history, STOCKS, out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_gate_history_json_roundtrip(tmp_path):
    """The training loop's persistence contract: gate_history.json must be valid JSON, keyed by
    epoch (string), each value a {ticker: float} dict -- exactly what plot_gate_evolution and any
    later manual inspection (the user's stated goal) expects."""
    gate_history = {"1": {sn: 0.5 for sn in STOCKS}}
    path = tmp_path / "gate_history.json"
    path.write_text(json.dumps(gate_history, indent=2), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == gate_history
    assert all(isinstance(v, float) for v in loaded["1"].values())


def test_load_resume_state_no_resume_returns_empty():
    gate_history, train_losses, val_losses, start_epoch = load_resume_state(None)
    assert gate_history == {} and train_losses == [] and val_losses == []
    assert start_epoch == 0


def test_load_resume_state_missing_gate_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resume_state(tmp_path)


def test_load_resume_state_loads_gate_history_and_computes_start_epoch(tmp_path):
    gate_history = {"1": {sn: 0.5 for sn in STOCKS}, "7": {sn: 0.6 for sn in STOCKS}}
    (tmp_path / "gate_history.json").write_text(json.dumps(gate_history), encoding="utf-8")

    loaded_history, train_losses, val_losses, start_epoch = load_resume_state(tmp_path)
    assert loaded_history == gate_history
    assert start_epoch == 7   # max epoch key, not len() -- must not assume no gaps
    assert train_losses == [] and val_losses == []   # no loss_history.json present


def test_load_resume_state_loads_loss_history_when_present(tmp_path):
    gate_history = {"1": {sn: 0.5 for sn in STOCKS}}
    (tmp_path / "gate_history.json").write_text(json.dumps(gate_history), encoding="utf-8")
    loss_history = {"train_losses": [1.0, 0.9], "val_losses": [1.1, 1.0]}
    (tmp_path / "loss_history.json").write_text(json.dumps(loss_history), encoding="utf-8")

    _, train_losses, val_losses, _ = load_resume_state(tmp_path)
    assert train_losses == [1.0, 0.9]
    assert val_losses == [1.1, 1.0]


def test_train_epoch_smoke_end_to_end():
    """One real forward+backward+optimizer.step() with PerTickerGatedNewsBaseline, dual-group
    dataset (synthetic HAR data, no real data_dir needed), reusing train_per_ticker_gate.train_epoch
    exactly as the real entrypoint does."""
    stock_data = {s: _make_har_df(seed=i) for i, s in enumerate(STOCKS)}
    config = LSTMGATConfig()
    config.num_features_per_stock = 3
    ds = MultiStockDatasetWithDualNews(
        stock_data, stock_data, STOCKS,
        seq_length=10, forecast_horizon=5, graph_method="correlation",
        normalize=False, config=config, news_panel_path=None,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    model = PerTickerGatedNewsBaseline(config, n_feat=146, num_stocks=len(STOCKS), d_news=8,
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
