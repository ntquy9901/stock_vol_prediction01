"""Coverage for the corr+lift ablation RUNNER without any GPU training.

``run_training`` is exercised on a tiny real HNX slice with ``train_masked_rich`` monkeypatched to a
deterministic stub, so the metric/DM/fit-evidence plumbing runs on CPU in a fraction of a second (no epochs,
no GPU). ``corrlift_adj_for`` is tested against REAL HNX close prices on a tiny slice. UNIQUE basename
(test_corrlift_runner.py) to avoid the pytest prepend-import duplicate-basename shadowing.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_corrlift_ablation as RCA  # noqa: E402
from config import SMOKE, Config     # noqa: E402

PANEL = "hnx"


def _tiny_D(n_keep=8):
    keep = RCA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:n_keep])
    td = tempfile.mkdtemp()
    D, _ = RCA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
    return D


def _stub_adj(N):
    diag = {"n_nodes": N, "n_pairs": N * (N - 1) // 2, "n_corr_edges": 1, "n_lift_edges": 0,
            "n_either_edges": 1, "n_both_edges": 0, "avg_off_degree": 0.1, "max_off_degree": 1,
            "n_singletons": 0, "n_train_rows": 500}
    return np.eye(N, dtype=np.float32), diag


# ------------------------- device label -------------------------

def test_device_label_both_arcs(monkeypatch):
    monkeypatch.setattr(RCA.torch.cuda, "is_available", lambda: True)
    assert RCA._device_label() == "gpu"
    monkeypatch.setattr(RCA.torch.cuda, "is_available", lambda: False)
    assert RCA._device_label() == "cpu"


# ------------------------- corrlift_adj_for on REAL HNX prices -------------------------

def test_corrlift_adj_for_real_slice():
    D = _tiny_D()
    adj, diag = RCA.corrlift_adj_for(D, str(RCA.VE.PRICE[PANEL]))
    assert adj.shape == (D.N, D.N)
    assert adj.dtype == np.float32
    assert np.allclose(np.diag(adj), 1.0)             # self-loop
    assert np.allclose(adj, adj.T)                    # undirected
    assert np.isfinite(adj).all()
    assert {"n_corr_edges", "n_lift_edges", "n_either_edges", "avg_off_degree"} <= set(diag)
    assert diag["n_nodes"] == D.N


def test_corrlift_adj_for_empty_val_raises():
    class _D:
        d_va = []
        tickers = ["AAA"]
    with pytest.raises(RuntimeError):
        RCA.corrlift_adj_for(_D, "ignored")


# ------------------------- run_training (stubbed train) -------------------------

def _patch_stub(monkeypatch, D):
    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        off = 0.0 if not use_graph else (1e-4 if adj is Dd.adj_vol2pk else 2e-4)
        test = Dd.y_te + off
        if return_splits:
            return {"test": test, "val": Dd.y_va + off, "train": Dd.y_tr + off,
                    "train_curve": [1e-6], "val_curve": [1.1e-6], "best_epoch": 1}
        return test
    monkeypatch.setattr(RCA.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(RCA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, []))
    monkeypatch.setattr(RCA, "corrlift_adj_for", lambda Dd, price_dir: _stub_adj(Dd.N))


def test_run_training_stubbed(monkeypatch, tmp_path):
    D = _tiny_D()
    _patch_stub(monkeypatch, D)
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    res = RCA.run_training(PANEL, cfg, horizon=1, out_dir=str(tmp_path))
    assert set(res["metrics_ensemble"]) == {"corr_lift_GAT", "stat_GAT_vol2pk", "no_graph_LSTM"}
    assert res["num_nodes"] == D.N
    assert res["device"] in {"cpu", "gpu"}
    assert res["design"] == "corrlift-graph-edge-ablation"
    assert set(res["dm"]) == {"corr_lift_vs_no_graph", "corr_lift_vs_stat", "stat_vs_no_graph"}
    assert res["edge_density"]["n_nodes"] == D.N
    assert (tmp_path / "corrlift_ablation_hnx_h1.json").exists()
    for m in res["metrics_ensemble"].values():
        assert np.isfinite(m["qlike"])
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)


def test_run_training_no_out_dir(monkeypatch):
    D = _tiny_D()
    _patch_stub(monkeypatch, D)
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    res = RCA.run_training(PANEL, cfg, horizon=1, out_dir=None)   # exercise the out_dir=None arc
    assert "metrics_ensemble" in res


def test_run_training_emits_overfit_evidence(monkeypatch, tmp_path):
    """run_training must stamp train/val metrics + per-model fit verdict + per-seed learning curves
    (CLAUDE.md over/under-fit mandate 2026-08-29) so the result.json can PROVE generalisation."""
    D = _tiny_D()

    def fake_train(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        goff = 0.0 if not use_graph else (1e-7 if adj is Dd.adj_vol2pk else 2e-7)
        test = Dd.y_te + 1e-5 + goff
        if return_splits:
            return {"test": test, "val": Dd.y_va + 5e-6 + goff, "train": Dd.y_tr + 1e-6 + goff,
                    "train_curve": [1e-6, 5e-7], "val_curve": [1.1e-6, 6e-7], "best_epoch": 2}
        return test

    monkeypatch.setattr(RCA.RMR, "train_masked_rich", fake_train)
    monkeypatch.setattr(RCA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, []))
    monkeypatch.setattr(RCA, "corrlift_adj_for", lambda Dd, price_dir: _stub_adj(Dd.N))
    cfg = Config(seeds=(42, 123), epochs=2, min_epochs=1, patience=1)
    res = RCA.run_training(PANEL, cfg, horizon=1, out_dir=str(tmp_path))
    configs = {"corr_lift_GAT", "stat_GAT_vol2pk", "no_graph_LSTM"}
    for block in ("train_metrics", "val_metrics", "fit_diagnostics", "learning_curves"):
        assert block in res and set(res[block]) == configs, block
    for k in configs:
        assert {"qlike", "r2"} <= set(res["train_metrics"][k])
        assert res["fit_diagnostics"][k]["status"] in {"ok", "overfit", "underfit", "unknown"}
        lc = res["learning_curves"][k]
        assert len(lc["train"]) == 2 and len(lc["val"]) == 2 and len(lc["best_epoch"]) == 2
        assert isinstance(lc["train"][0], list)
        # stub error grows train<val<test -> correct split wiring yields this MSE ordering
        assert res["train_metrics"][k]["mse"] < res["val_metrics"][k]["mse"] < res["metrics_ensemble"][k]["mse"]


# ------------------------- forward_pass_smoke + build_panel_masked guards -------------------------

def test_forward_pass_smoke_empty_test_raises():
    class _D:
        X_te = np.zeros((0, 3, 4, 5), dtype=np.float32)
        nmask_te = np.zeros((0, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RCA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), batch=2)


def test_forward_pass_non_finite_raises(monkeypatch):
    import torch

    class _Net:
        def eval(self):
            return self

        def __call__(self, xb, adj_b):
            return torch.full((xb.shape[0], xb.shape[1]), float("nan"))

    monkeypatch.setattr(RCA.RMR, "MaskedRichNet", lambda *a, **k: _Net())

    class _D:
        X_te = np.zeros((2, 3, 4, 5), dtype=np.float32)
        nmask_te = np.ones((2, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RCA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), batch=2)


def test_build_panel_masked_too_few_raises(monkeypatch):
    monkeypatch.setattr(RCA.EFA, "_write_estimator_processed", lambda *a, **k: ["one.csv"])
    with pytest.raises(RuntimeError):
        RCA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir="x", keep_tickers={"AAA"})


# ------------------------- main() dry + train branches -------------------------

def test_main_dry_branch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_corrlift_ablation.py", "--panel", PANEL,
                                      "--horizon", "1", "--max-tickers", "8"])
    RCA.main()
    assert "forward pass OK" in capsys.readouterr().out


def test_run_dry_no_cap(monkeypatch, capsys):
    """max_tickers=0 exercises the 'no cap' arc of run_dry (heavy calls stubbed for speed)."""
    D = _tiny_D()
    monkeypatch.setattr(RCA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, []))
    monkeypatch.setattr(RCA, "corrlift_adj_for", lambda Dd, price_dir: _stub_adj(Dd.N))
    monkeypatch.setattr(RCA, "forward_pass_smoke",
                        lambda Dd, adj, batch=2: np.zeros((batch, Dd.N), dtype=np.float32))
    res = RCA.run_dry(PANEL, horizon=1, max_tickers=0)
    assert res["n_nodes"] == D.N
    assert "forward pass OK" in capsys.readouterr().out


def test_main_train_branch_stubbed(monkeypatch, capsys):
    captured = {}

    def fake_run_training(panel, cfg, horizon, out_dir=None):
        captured["epochs"] = cfg.epochs
        captured["seeds"] = cfg.seeds
        return {"n_test_obs": 10,
                "metrics_ensemble": {"corr_lift_GAT": {"mse": 1e-4, "rmse": 0.01, "mae": 0.008,
                                                       "qlike": 0.5, "r2": 0.1, "n": 10}},
                "dm": {"corr_lift_vs_no_graph": {"qlike": {"p_value": 0.3, "favors": "A", "mean_diff": -0.01}}}}

    monkeypatch.setattr(RCA, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["run_corrlift_ablation.py", "--panel", PANEL, "--horizon", "1",
                                      "--train-epochs", "10", "--seeds", "42", "123", "2026"])
    RCA.main()
    assert captured["epochs"] == 10 and captured["seeds"] == (42, 123, 2026)
    out = capsys.readouterr().out
    assert "QLIKE" in out and "DM corr_lift_vs_no_graph" in out
