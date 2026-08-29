"""Coverage for the DY ablation runner WITHOUT any GPU training.

``run_training`` is exercised on a tiny real HNX slice with ``train_masked_rich`` monkeypatched to a
deterministic stub, so the metric/DM plumbing (and the DY-adjacency build) run on CPU in a fraction of a
second (no epochs, no GPU). The dry / main branches and the fail-loud guards are covered too.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_dy_ablation as RDA  # noqa: E402
from config import SMOKE, Config  # noqa: E402

PANEL = "hnx"


def _tiny_D_and_files(n=8):
    keep = RDA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:n])
    td = tempfile.mkdtemp()
    D, files = RDA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
    return D, files


def test_run_training_stubbed(monkeypatch, tmp_path):
    D, files = _tiny_D_and_files()

    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        off = 0.0 if not use_graph else 1e-4
        return Dd.y_te + off

    monkeypatch.setattr(RDA.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(RDA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, files))
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    assert RDA.run_training(PANEL, cfg, horizon=1, out_dir=None)["num_nodes"] == D.N  # out_dir falsy branch
    res = RDA.run_training(PANEL, cfg, horizon=1, out_dir=str(tmp_path))
    assert set(res["metrics_ensemble"]) == {"dy_GAT", "stat_GAT_vol2pk", "no_graph_LSTM"}
    assert res["num_nodes"] == D.N
    assert res["device"] == "cpu"
    assert abs(res["dy_connectedness"]["row_sum_mean"] - 1.0) < 1e-5
    assert (tmp_path / "dy_ablation_hnx_h1.json").exists()
    for m in res["metrics_ensemble"].values():
        assert np.isfinite(m["qlike"])
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)
    assert set(res["dm"]) == {"dy_vs_no_graph", "dy_vs_stat", "stat_vs_no_graph"}


def test_load_sector_context_present_and_absent(monkeypatch, tmp_path):
    # absent -> None
    monkeypatch.setattr(RDA, "SECTOR_RESULT", tmp_path / "nope.json")
    assert RDA._load_sector_dm({}, 1, 1e-8) is None
    # present -> surfaces the recorded sector-GAT qlike
    import json
    f = tmp_path / "sector_ablation_hnx_h1.json"
    f.write_text(json.dumps({"metrics_ensemble": {"sector_GAT": {"qlike": 1.9, "mse": 1e-6,
                                                                 "rmse": 1e-3, "mae": 1e-3, "r2": 0.2}}}))
    monkeypatch.setattr(RDA, "SECTOR_RESULT", f)
    ctx = RDA._load_sector_dm({}, 1, 1e-8)
    assert ctx["sector_GAT_qlike"] == 1.9


def test_main_dry_branch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_dy_ablation.py", "--panel", PANEL,
                                      "--horizon", "1", "--max-tickers", "8"])
    RDA.main()
    out = capsys.readouterr().out
    assert "forward pass OK" in out and "total-connectedness" in out


def test_run_dry_no_cap_branch(monkeypatch, capsys):
    """run_dry with max_tickers=0 exercises the 'no cap' branch (keep universe unchanged) using stubs so
    no real full-universe panel build runs."""
    class _D:
        N = 3
        X_te = np.zeros((2, 3, 4, 5), dtype=np.float32)
        nmask_te = np.ones((2, 3), dtype=bool)
    monkeypatch.setattr(RDA.EFA, "screened_tickers", lambda panel: {"A", "B", "C"})
    monkeypatch.setattr(RDA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (_D(), []))
    adj = np.eye(3, dtype=np.float32)
    monkeypatch.setattr(RDA, "dy_adj_for", lambda D, files, **k: (adj, {
        "total_connectedness_index": 5.0, "row_sum_mean": 1.0, "diag_mean_own_share": 0.9,
        "asymmetry_frob": 0.1}))
    monkeypatch.setattr(RDA, "forward_pass_smoke", lambda D, a, batch=2: np.zeros((2, 3)))
    out = RDA.run_dry(PANEL, 1, max_tickers=0)
    assert out["n_nodes"] == 3
    assert "forward pass OK" in capsys.readouterr().out


def test_main_train_branch_stubbed(monkeypatch, capsys):
    captured = {}

    def fake_run_training(panel, cfg, horizon, out_dir=None, p=1, H=10, alpha=0.05, l1_ratio=0.5):
        captured["epochs"] = cfg.epochs
        captured["seeds"] = cfg.seeds
        captured["p"] = p
        return {"n_test_obs": 10,
                "metrics_ensemble": {"dy_GAT": {"mse": 1e-4, "rmse": 0.01, "mae": 0.008,
                                                "qlike": 0.5, "r2": 0.1, "n": 10}},
                "dm": {"dy_vs_no_graph": {"qlike": {"p_value": 0.3, "favors": "A", "mean_diff": -0.01}}}}

    monkeypatch.setattr(RDA, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["run_dy_ablation.py", "--panel", PANEL, "--horizon", "1",
                                      "--train-epochs", "10", "--seeds", "42", "123", "2026",
                                      "--var-lag", "2"])
    RDA.main()
    assert captured["epochs"] == 10 and captured["seeds"] == (42, 123, 2026) and captured["p"] == 2
    out = capsys.readouterr().out
    assert "QLIKE" in out and "DM dy_vs_no_graph" in out


def test_forward_pass_smoke_empty_test_raises():
    class _D:
        X_te = np.zeros((0, 3, 4, 5), dtype=np.float32)
        nmask_te = np.zeros((0, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RDA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), batch=2)


def test_forward_pass_non_finite_raises(monkeypatch):
    import torch

    class _Net:
        def eval(self):
            return self

        def __call__(self, xb, adj_b):
            return torch.full((xb.shape[0], xb.shape[1]), float("nan"))

    monkeypatch.setattr(RDA.RMR, "MaskedRichNet", lambda *a, **k: _Net())

    class _D:
        X_te = np.zeros((2, 3, 4, 5), dtype=np.float32)
        nmask_te = np.ones((2, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RDA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), batch=2)


def test_build_panel_masked_too_few_raises(monkeypatch):
    monkeypatch.setattr(RDA.EFA, "_write_estimator_processed", lambda *a, **k: ["one.csv"])
    with pytest.raises(RuntimeError):
        RDA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir="x", keep_tickers={"AAA"})


def test_dy_adj_for_shape_mismatch_raises(monkeypatch):
    class _D:
        tickers = ["A", "B", "C"]
        N = 3
        d_va = ["2020-01-01"]
    monkeypatch.setattr(RDA.DY, "train_vol_panel", lambda files, tk, vs: np.zeros((50, 3)))
    monkeypatch.setattr(RDA.DY, "build_dy_adjacency",
                        lambda *a, **k: (np.eye(2, dtype=np.float32), np.eye(2), {}))
    with pytest.raises(RuntimeError):
        RDA.dy_adj_for(_D, [])
