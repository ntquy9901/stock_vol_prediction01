"""Coverage for the fetch helpers and the panel-agnostic ablation runner WITHOUT any GPU training.

``run_training`` is exercised on a tiny real HNX slice with ``train_masked_rich`` monkeypatched to a
deterministic stub, so the metric/DM plumbing runs on CPU in a fraction of a second (no epochs, no GPU).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import fetch_sectors as FS            # noqa: E402  (sp500 GICS fetch)
import fetch_vn_sectors as FV         # noqa: E402  (VN ICB fetch)
import run_sector_ablation as RSA     # noqa: E402
from config import SMOKE, Config      # noqa: E402

PANEL = "hnx"


# ------------------------- fetch_sectors (sp500 GICS) -------------------------

def test_sanitize_dot_to_dash():
    assert FS.sanitize("BRK.B") == "BRK-B"
    assert FS.sanitize(" BF.B ") == "BF-B"


def test_build_sector_map_dashes_and_skips_blank():
    df = pd.DataFrame({"Symbol": ["AAPL", "BRK.B", "XXX"],
                       "GICS Sector": ["Information Technology", "Financials", None]})
    assert FS.build_sector_map(df) == {"AAPL": "Information Technology", "BRK-B": "Financials"}


def test_build_sector_map_missing_column_raises():
    with pytest.raises(ValueError):
        FS.build_sector_map(pd.DataFrame({"Symbol": ["AAPL"]}))


def test_sp500_write_and_load_roundtrip(tmp_path):
    from sector_adjacency import load_sector_map
    m = {"AAPL": "Information Technology", "JPM": "Financials"}
    out = FS.write_sector_csv(m, tmp_path / "s.csv", "http://src", "2026-08-29")
    assert load_sector_map(out) == m


def test_sp500_fetch_main_writes_csv(tmp_path, monkeypatch):
    fake = pd.DataFrame({"Symbol": ["AAPL", "JPM"],
                         "GICS Sector": ["Information Technology", "Financials"]})
    monkeypatch.setattr(FS.pd, "read_csv", lambda *a, **k: fake)
    out = tmp_path / "s.csv"
    monkeypatch.setattr(sys, "argv", ["fetch_sectors.py", "--fetched-date", "2026-08-29", "--out", str(out)])
    FS.main()
    assert out.exists() and len(pd.read_csv(out)) == 2


# ------------------------- fetch_vn_sectors (ICB) -------------------------

def test_vn_build_sector_map_upper_and_skip_blank():
    df = pd.DataFrame({"symbol": ["aaa", "shb", "zzz"],
                       "industry_name": ["Nhua", "Ngan hang", None]})
    assert FV.build_vn_sector_map(df) == {"AAA": "Nhua", "SHB": "Ngan hang"}


def test_vn_build_sector_map_by_code_level():
    df = pd.DataFrame({"symbol": ["AAA"], "industry_code": [8600], "industry_name": ["x"]})
    assert FV.build_vn_sector_map(df, level="industry_code") == {"AAA": "8600"}


def test_vn_build_sector_map_missing_column_raises():
    with pytest.raises(ValueError):
        FV.build_vn_sector_map(pd.DataFrame({"symbol": ["AAA"]}))


def test_vn_main_from_raw(tmp_path, monkeypatch):
    from sector_adjacency import load_sector_map
    raw = tmp_path / "raw.csv"
    pd.DataFrame({"symbol": ["AAA", "SHB"], "industry_code": [1, 2],
                  "industry_name": ["Nhua", "Ngan hang"]}).to_csv(raw, index=False)
    out = tmp_path / "vn.csv"
    monkeypatch.setattr(sys, "argv", ["fetch_vn_sectors.py", "--from-raw", str(raw),
                                      "--fetched-date", "2026-08-29", "--out", str(out)])
    FV.main()
    assert load_sector_map(out) == {"AAA": "Nhua", "SHB": "Ngan hang"}


def test_vn_main_requires_source(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fetch_vn_sectors.py", "--fetched-date", "2026-08-29"])
    with pytest.raises(SystemExit):
        FV.main()


# ------------------------- runner -------------------------

def _tiny_D():
    keep = RSA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:8])
    td = tempfile.mkdtemp()
    D, _ = RSA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
    return D


def test_default_sector_csv_routing():
    assert RSA.default_sector_csv("sp500").name == "sp500_gics_sectors.csv"
    assert RSA.default_sector_csv("hnx").name == "vn_sectors.csv"
    assert RSA.default_sector_csv("vn100").name == "vn_sectors.csv"


def test_run_training_stubbed(monkeypatch, tmp_path):
    D = _tiny_D()

    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        off = 0.0 if not use_graph else (1e-4 if adj is Dd.adj_vol2pk else 2e-4)
        return Dd.y_te + off

    monkeypatch.setattr(RSA.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(RSA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, []))
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    res = RSA.run_training(PANEL, cfg, horizon=1, out_dir=str(tmp_path))
    assert set(res["metrics_ensemble"]) == {"sector_GAT", "stat_GAT_vol2pk", "no_graph_LSTM"}
    assert res["num_nodes"] == D.N
    assert res["device"] == "cpu"
    assert (tmp_path / "sector_ablation_hnx_h1.json").exists()
    for m in res["metrics_ensemble"].values():
        assert np.isfinite(m["qlike"])
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)


def test_main_dry_branch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_sector_ablation.py", "--panel", PANEL,
                                      "--horizon", "1", "--max-tickers", "8"])
    RSA.main()
    assert "forward pass OK" in capsys.readouterr().out


def test_main_train_branch_stubbed(monkeypatch, capsys):
    captured = {}

    def fake_run_training(panel, cfg, horizon, sector_csv=None, out_dir=None):
        captured["epochs"] = cfg.epochs
        captured["seeds"] = cfg.seeds
        return {"n_test_obs": 10,
                "metrics_ensemble": {"sector_GAT": {"mse": 1e-4, "rmse": 0.01, "mae": 0.008,
                                                    "qlike": 0.5, "r2": 0.1, "n": 10}},
                "dm": {"sector_vs_stat": {"qlike": {"p_value": 0.3, "favors": "A", "mean_diff": -0.01}}}}

    monkeypatch.setattr(RSA, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["run_sector_ablation.py", "--panel", PANEL, "--horizon", "1",
                                      "--train-epochs", "8", "--seeds", "42", "123"])
    RSA.main()
    assert captured["epochs"] == 8 and captured["seeds"] == (42, 123)
    out = capsys.readouterr().out
    assert "QLIKE" in out and "DM sector_vs_stat" in out


def test_forward_pass_smoke_empty_test_raises():
    class _D:
        X_te = np.zeros((0, 3, 4, 5), dtype=np.float32)
        nmask_te = np.zeros((0, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RSA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), batch=2)


def test_forward_pass_non_finite_raises(monkeypatch):
    import torch

    class _Net:
        def eval(self):
            return self

        def __call__(self, xb, adj_b):
            return torch.full((xb.shape[0], xb.shape[1]), float("nan"))

    monkeypatch.setattr(RSA.RMR, "MaskedRichNet", lambda *a, **k: _Net())

    class _D:
        X_te = np.zeros((2, 3, 4, 5), dtype=np.float32)
        nmask_te = np.ones((2, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RSA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), batch=2)


def test_build_panel_masked_too_few_raises(monkeypatch):
    monkeypatch.setattr(RSA.EFA, "_write_estimator_processed", lambda *a, **k: ["one.csv"])
    with pytest.raises(RuntimeError):
        RSA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir="x", keep_tickers={"AAA"})
