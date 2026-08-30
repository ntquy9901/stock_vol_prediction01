"""Coverage for the hetero ablation RUNNER + train_hetero_rich WITHOUT heavy GPU training.

``run_training`` runs on a tiny real HNX slice with the train fns monkeypatched to deterministic stubs (metric
/ DM / fit-evidence plumbing on CPU in a fraction of a second). ``train_hetero_rich`` gets ONE real tiny 1-epoch
CPU run (real-data smoke) plus a constant-net multi-epoch run for the early-stop branch. UNIQUE basename
(test_hetero_runner.py) to avoid the pytest prepend-import duplicate-basename shadowing.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_hetero_ablation as RHA  # noqa: E402
import hetero_model as HM          # noqa: E402
from config import SMOKE, Config   # noqa: E402

PANEL = "hnx"


def _tiny_D(n_keep=8):
    keep = RHA.EFA.screened_tickers(PANEL)
    if keep is None:
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(keep)[:n_keep])
    td = tempfile.mkdtemp()
    D, _ = RHA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir=td, keep_tickers=keep)
    return D


@pytest.fixture(scope="module")
def tiny_D():
    return _tiny_D()


def _stub_diag(N):
    return {"n_nodes": N, "n_pairs": N * (N - 1) // 2,
            "linear_corr": {"thresh": 0.25, "n_edges": 1, "avg_off_degree": 0.1, "max_off_degree": 1,
                            "n_singletons": 0, "minmax": [0.3, 0.9]},
            "nonlinear_assoc": {"thresh": 1.2, "n_edges": 1, "avg_off_degree": 0.1, "max_off_degree": 1,
                                "n_singletons": 0, "minmax": [1.3, 2.0]},
            "n_both_relations_edges": 0, "n_train_rows": 500}


# ------------------------- device label -------------------------

def test_device_label_both_arcs(monkeypatch):
    monkeypatch.setattr(RHA.torch.cuda, "is_available", lambda: True)
    assert RHA._device_label() == "gpu"
    monkeypatch.setattr(RHA.torch.cuda, "is_available", lambda: False)
    assert RHA._device_label() == "cpu"


# ------------------------- hetero_adj_for on REAL HNX prices -------------------------

def test_hetero_adj_for_real_slice(tiny_D):
    adj_lin, adj_nl, adj_sq, diag = RHA.hetero_adj_for(tiny_D, str(RHA.VE.PRICE[PANEL]))
    for adj in (adj_lin, adj_nl, adj_sq):
        assert adj.shape == (tiny_D.N, tiny_D.N)
        assert adj.dtype == np.float32
        assert np.allclose(np.diag(adj), 1.0)
        assert np.isfinite(adj).all()
    assert np.allclose(adj_lin, adj_lin.T) and np.allclose(adj_nl, adj_nl.T)
    assert {"linear_corr", "nonlinear_assoc", "squashed_lowered"} <= set(diag)
    assert diag["n_nodes"] == tiny_D.N


def test_hetero_adj_for_empty_val_raises():
    class _D:
        d_va = []
        tickers = ["AAA"]
    with pytest.raises(RuntimeError):
        RHA.hetero_adj_for(_D, "ignored")


# ------------------------- run_training (stubbed train) -------------------------

def _patch_stub(monkeypatch, D):
    def fake_hetero(Dd, cfg, seed, adj_lin, adj_nl, return_splits=False):
        test = Dd.y_te + 2e-4
        if return_splits:
            return {"test": test, "val": Dd.y_va + 2e-4, "train": Dd.y_tr + 2e-4,
                    "train_curve": [1e-6], "val_curve": [1.1e-6], "best_epoch": 1}
        return test

    def fake_masked(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        off = 0.0 if not use_graph else 1e-4
        test = Dd.y_te + off
        if return_splits:
            return {"test": test, "val": Dd.y_va + off, "train": Dd.y_tr + off,
                    "train_curve": [1e-6], "val_curve": [1.1e-6], "best_epoch": 1}
        return test

    monkeypatch.setattr(RHA.HM, "train_hetero_rich", fake_hetero)
    monkeypatch.setattr(RHA.RMR, "train_masked_rich", fake_masked)
    monkeypatch.setattr(RHA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (D, []))
    monkeypatch.setattr(RHA, "hetero_adj_for",
                        lambda Dd, price_dir: (np.eye(Dd.N, dtype=np.float32),
                                               np.eye(Dd.N, dtype=np.float32),
                                               np.eye(Dd.N, dtype=np.float32), _stub_diag(Dd.N)))


def test_run_training_stubbed(monkeypatch, tmp_path, tiny_D):
    _patch_stub(monkeypatch, tiny_D)
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    res = RHA.run_training(PANEL, cfg, horizon=1, out_dir=str(tmp_path))
    assert set(res["metrics_ensemble"]) == {"hetero_2rel_GAT", "squashed_lowered_GAT", "no_graph_LSTM"}
    assert res["num_nodes"] == tiny_D.N
    assert res["device"] in {"cpu", "gpu"}
    assert res["design"] == "hetero-2relation-graph-edge-ablation"
    assert res["aggregation"] == "sum"
    assert set(res["dm"]) == {"hetero_vs_no_graph", "hetero_vs_squashed_lowered",
                              "squashed_lowered_vs_no_graph"}
    assert res["edge_density"]["n_nodes"] == tiny_D.N
    assert res["prior_squashed_paper_qlike"] == RHA.PRIOR_SQUASHED_PAPER_QLIKE
    assert (tmp_path / "hetero_graph_ablation_hnx_h1.json").exists()
    for m in res["metrics_ensemble"].values():
        assert np.isfinite(m["qlike"])
        assert {"mse", "rmse", "mae", "qlike", "r2"} <= set(m)


def test_run_training_no_out_dir(monkeypatch, tiny_D):
    _patch_stub(monkeypatch, tiny_D)
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1)
    res = RHA.run_training(PANEL, cfg, horizon=1, out_dir=None)
    assert "metrics_ensemble" in res


def test_run_training_emits_overfit_evidence(monkeypatch, tmp_path, tiny_D):
    def fake_hetero(Dd, cfg, seed, adj_lin, adj_nl, return_splits=False):
        test = Dd.y_te + 1e-5 + 2e-7
        if return_splits:
            return {"test": test, "val": Dd.y_va + 5e-6 + 2e-7, "train": Dd.y_tr + 1e-6 + 2e-7,
                    "train_curve": [1e-6, 5e-7], "val_curve": [1.1e-6, 6e-7], "best_epoch": 2}
        return test

    def fake_masked(Dd, cfg, seed, use_graph, adj, output_param="zscore_floor", return_splits=False):
        goff = 0.0 if not use_graph else 1e-7
        test = Dd.y_te + 1e-5 + goff
        if return_splits:
            return {"test": test, "val": Dd.y_va + 5e-6 + goff, "train": Dd.y_tr + 1e-6 + goff,
                    "train_curve": [1e-6, 5e-7], "val_curve": [1.1e-6, 6e-7], "best_epoch": 2}
        return test

    monkeypatch.setattr(RHA.HM, "train_hetero_rich", fake_hetero)
    monkeypatch.setattr(RHA.RMR, "train_masked_rich", fake_masked)
    monkeypatch.setattr(RHA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (tiny_D, []))
    monkeypatch.setattr(RHA, "hetero_adj_for",
                        lambda Dd, price_dir: (np.eye(Dd.N, dtype=np.float32),
                                               np.eye(Dd.N, dtype=np.float32),
                                               np.eye(Dd.N, dtype=np.float32), _stub_diag(Dd.N)))
    cfg = Config(seeds=(42, 123), epochs=2, min_epochs=1, patience=1)
    res = RHA.run_training(PANEL, cfg, horizon=1, out_dir=str(tmp_path))
    configs = {"hetero_2rel_GAT", "squashed_lowered_GAT", "no_graph_LSTM"}
    for block in ("train_metrics", "val_metrics", "fit_diagnostics", "learning_curves"):
        assert block in res and set(res[block]) == configs, block
    for k in configs:
        assert {"qlike", "r2"} <= set(res["train_metrics"][k])
        assert res["fit_diagnostics"][k]["status"] in {"ok", "overfit", "underfit", "unknown"}
        lc = res["learning_curves"][k]
        assert len(lc["train"]) == 2 and len(lc["val"]) == 2 and len(lc["best_epoch"]) == 2
        assert isinstance(lc["train"][0], list)
        assert res["train_metrics"][k]["mse"] < res["val_metrics"][k]["mse"] < res["metrics_ensemble"][k]["mse"]


# ------------------------- train_hetero_rich (real tiny + branch coverage) -------------------------

def test_train_hetero_rich_real_tiny_cpu(monkeypatch, tiny_D):
    """One real 1-epoch CPU run of the actual training loop (real-data smoke)."""
    monkeypatch.setattr(HM.torch.cuda, "is_available", lambda: False)  # deterministic CPU
    adj = np.eye(tiny_D.N, dtype=np.float32)
    cfg = Config(seeds=(42,), epochs=1, min_epochs=1, patience=1, batch_size=8, hidden=8, heads=2)
    out = HM.train_hetero_rich(tiny_D, cfg, 42, adj, adj, return_splits=True)
    assert set(out) == {"test", "val", "train", "train_curve", "val_curve", "best_epoch"}
    assert out["test"].shape == tiny_D.y_te.shape
    assert np.isfinite(out["test"]).all()
    assert (out["test"] > 0).all()                          # positivity floor applied
    assert len(out["train_curve"]) == 1


def test_train_hetero_rich_earlystop_and_return_array(monkeypatch, tiny_D):
    """Constant-output net -> val MSE identical every epoch -> covers the no-improve (else) + early-stop break
    branches, and the return_splits=False (return array) path."""
    class _ConstNet(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x, al, an):
            b, n = x.shape[0], x.shape[1]
            return (self.p * 0.0).reshape(1, 1).expand(b, n)

    monkeypatch.setattr(HM.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(HM, "HeteroRichNet", _ConstNet)
    adj = np.eye(tiny_D.N, dtype=np.float32)
    cfg = Config(seeds=(7,), epochs=3, min_epochs=1, patience=1, batch_size=8)
    te = HM.train_hetero_rich(tiny_D, cfg, 7, adj, adj, return_splits=False)
    assert te.shape == tiny_D.y_te.shape
    assert np.isfinite(te).all()


def test_train_hetero_rich_zero_epochs_no_best_state(monkeypatch, tiny_D):
    """epochs=0 -> the train loop never runs -> best_state stays None (covers the 'if best_state' False arc):
    inference falls back to the freshly-initialised net."""
    monkeypatch.setattr(HM.torch.cuda, "is_available", lambda: False)
    adj = np.eye(tiny_D.N, dtype=np.float32)
    cfg = Config(seeds=(1,), epochs=0, min_epochs=1, patience=1, batch_size=8, hidden=8, heads=2)
    out = HM.train_hetero_rich(tiny_D, cfg, 1, adj, adj, return_splits=True)
    assert out["best_epoch"] == 0 and out["train_curve"] == []
    assert np.isfinite(out["test"]).all()


# ------------------------- forward_pass_smoke + build_panel_masked guards -------------------------

def test_forward_pass_smoke_empty_test_raises():
    class _D:
        X_te = np.zeros((0, 3, 4, 5), dtype=np.float32)
        nmask_te = np.zeros((0, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RHA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), batch=2)


def test_forward_pass_non_finite_raises(monkeypatch):
    class _Net:
        def eval(self):
            return self

        def __call__(self, xb, al, an):
            return torch.full((xb.shape[0], xb.shape[1]), float("nan"))

    monkeypatch.setattr(RHA.HM, "HeteroRichNet", lambda *a, **k: _Net())

    class _D:
        X_te = np.zeros((2, 3, 4, 5), dtype=np.float32)
        nmask_te = np.ones((2, 3), dtype=bool)
        N = 3
    with pytest.raises(RuntimeError):
        RHA.forward_pass_smoke(_D, np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32), batch=2)


def test_build_panel_masked_too_few_raises(monkeypatch):
    monkeypatch.setattr(RHA.EFA, "_write_estimator_processed", lambda *a, **k: ["one.csv"])
    with pytest.raises(RuntimeError):
        RHA.build_panel_masked(PANEL, SMOKE, horizon=1, out_dir="x", keep_tickers={"AAA"})


# ------------------------- main() dry + train branches -------------------------

def test_main_dry_branch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_hetero_ablation.py", "--panel", PANEL,
                                      "--horizon", "1", "--max-tickers", "8"])
    RHA.main()
    assert "forward pass OK" in capsys.readouterr().out


def test_run_dry_no_cap(monkeypatch, capsys, tiny_D):
    monkeypatch.setattr(RHA, "build_panel_masked",
                        lambda panel, cfg, horizon, out_dir, keep_tickers=None: (tiny_D, []))
    monkeypatch.setattr(RHA, "hetero_adj_for",
                        lambda Dd, price_dir: (np.eye(Dd.N, dtype=np.float32),
                                               np.eye(Dd.N, dtype=np.float32),
                                               np.eye(Dd.N, dtype=np.float32), _stub_diag(Dd.N)))
    monkeypatch.setattr(RHA, "forward_pass_smoke",
                        lambda Dd, al, an, batch=2: np.zeros((batch, Dd.N), dtype=np.float32))
    res = RHA.run_dry(PANEL, horizon=1, max_tickers=0)
    assert res["n_nodes"] == tiny_D.N
    assert "forward pass OK" in capsys.readouterr().out


def test_main_train_branch_stubbed(monkeypatch, capsys):
    captured = {}

    def fake_run_training(panel, cfg, horizon, out_dir=None):
        captured["epochs"] = cfg.epochs
        captured["seeds"] = cfg.seeds
        return {"n_test_obs": 10,
                "metrics_ensemble": {"hetero_2rel_GAT": {"mse": 1e-4, "rmse": 0.01, "mae": 0.008,
                                                         "qlike": 0.5, "r2": 0.1, "n": 10}},
                "dm": {"hetero_vs_no_graph": {"qlike": {"p_value": 0.3, "favors": "A", "mean_diff": -0.01}}}}

    monkeypatch.setattr(RHA, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["run_hetero_ablation.py", "--panel", PANEL, "--horizon", "1",
                                      "--train-epochs", "10", "--seeds", "42", "123", "2026"])
    RHA.main()
    assert captured["epochs"] == 10 and captured["seeds"] == (42, 123, 2026)
    out = capsys.readouterr().out
    assert "QLIKE" in out and "DM hetero_vs_no_graph" in out
