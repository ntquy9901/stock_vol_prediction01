"""Smoke + property tests for the rich masked-panel graph experiment (pytest).

Covers the two new pieces:
  * ``WeightedGATLayer`` MUST consume edge weight AND sign (the whole point vs the binary GATLayer),
    and MUST stay mask-aware (a non-neighbour node cannot influence a target's output).
  * ``build_masked_rich`` produces a 5-feature masked panel + two square, self-looped adjacencies on a
    small real VN30 data slice (real-data smoke, per project rule).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "submission" / "soict_lstm_gat"))

import masked_rich as MR  # noqa: E402
from run_masked_rich import WeightedGATLayer, MaskedRichNet, train_masked_rich  # noqa: E402
from config import SMOKE  # noqa: E402

_VN30 = sorted(glob.glob(str(REPO / "submission" / "soict_lstm_gat" / "data" / "vn30" / "*_processed.csv")))
_PRICE = REPO / "data" / "raw" / "prices"


def _layer(seed=0):
    torch.manual_seed(seed)
    return WeightedGATLayer(in_dim=5, out_dim=8, heads=2).eval()


def test_wgat_consumes_edge_sign():
    """Flipping an edge's SIGN must change the output (binary GATLayer would not)."""
    lay = _layer()
    torch.manual_seed(1)
    h = torch.randn(1, 4, 5)
    A = torch.zeros(1, 4, 4)
    for i in range(4):
        A[0, i, i] = 1.0
    A[0, 0, 1] = 0.7                       # edge source 1 -> target 0
    out_pos = lay(h, A)
    A_neg = A.clone(); A_neg[0, 0, 1] = -0.7
    out_neg = lay(h, A_neg)
    assert not torch.allclose(out_pos[0, 0], out_neg[0, 0], atol=1e-6), "sign flip did not change output"


def test_wgat_consumes_edge_weight():
    """Changing an edge's WEIGHT magnitude must change the output."""
    lay = _layer()
    torch.manual_seed(2)
    h = torch.randn(1, 4, 5)
    A = torch.eye(4).unsqueeze(0).clone()
    A[0, 0, 1] = 0.2
    out_a = lay(h, A)
    A2 = A.clone(); A2[0, 0, 1] = 0.9
    out_b = lay(h, A2)
    assert not torch.allclose(out_a[0, 0], out_b[0, 0], atol=1e-6), "weight change did not change output"


def test_wgat_mask_aware():
    """A node that is NOT a neighbour of target 0 cannot influence target 0's output."""
    lay = _layer()
    torch.manual_seed(3)
    h = torch.randn(1, 4, 5)
    A = torch.eye(4).unsqueeze(0).clone()
    A[0, 0, 1] = 0.5                        # target 0's only non-self neighbour is node 1
    out1 = lay(h, A)
    h2 = h.clone(); h2[0, 3] = torch.randn(5)   # perturb node 3 (a non-neighbour of target 0)
    out2 = lay(h2, A)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6), "non-neighbour leaked into target output"


@pytest.mark.skipif(len(_VN30) < 5, reason="VN30 processed data not available")
def test_build_masked_rich_shapes():
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    assert D.X_tr.shape[-1] == MR.N_FEAT == 5
    assert D.X_tr.shape[1] == D.N and D.adj_vol2pk.shape == (D.N, D.N)
    assert D.adj_corr.shape == (D.N, D.N)
    assert np.allclose(np.diag(D.adj_vol2pk), 1.0)         # self-loop
    assert np.allclose(np.diag(D.adj_corr), 1.0)
    assert D.har_tr.shape[-1] == 3                          # HAR baseline still 3-feat
    # masks: a scored target cell must also have a valid input window
    assert (D.tmask_te <= D.nmask_te).all()
    # vol->PK is DIRECTED (asymmetric) whereas corr edge is symmetric in support
    assert not np.allclose(D.adj_vol2pk, D.adj_vol2pk.T)
    # features finite after nan_to_num
    assert np.isfinite(D.X_tr).all() and np.isfinite(D.y_te).all()


@pytest.mark.skipif(len(_VN30) < 5, reason="VN30 processed data not available")
def test_train_masked_rich_smoke():
    """End-to-end: build + train (2 epochs, 1 seed) both no-graph and weighted-GAT paths run + return
    finite floored predictions of the right shape."""
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    p_lstm = train_masked_rich(D, SMOKE, seed=42, use_graph=False, adj=D.adj_vol2pk)
    p_gat = train_masked_rich(D, SMOKE, seed=42, use_graph=True, adj=D.adj_vol2pk)
    assert p_lstm.shape == D.y_te.shape == p_gat.shape
    assert np.isfinite(p_gat).all() and (p_gat > 0).all()


@pytest.mark.skipif(len(_VN30) < 5, reason="VN30 processed data not available")
def test_train_masked_rich_ratio_exp_positive():
    """output_param='ratio_exp' (node-scaled ratio target + exp output, no economic floor) yields strictly
    positive, finite predictions by construction; the default 'zscore_floor' path stays a distinct mapping."""
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    p_default = train_masked_rich(D, SMOKE, seed=42, use_graph=False, adj=D.adj_vol2pk)
    p_ratio = train_masked_rich(D, SMOKE, seed=42, use_graph=False, adj=D.adj_vol2pk, output_param="ratio_exp")
    assert p_ratio.shape == D.y_te.shape
    assert np.isfinite(p_ratio).all() and (p_ratio > 0).all()   # positive by construction, no floor needed
    assert not np.allclose(p_default, p_ratio)                   # genuinely different parameterizations


def test_masked_rich_net_forward():
    net = MaskedRichNet(hidden=8, heads=2, use_graph=True).eval()
    assert net.gat_layers == 2 and hasattr(net, "gat2")   # 2-hop default (matches deliverable)
    x = torch.randn(3, 6, 10, 5)
    adj = torch.eye(6).unsqueeze(0).expand(3, 6, 6).contiguous()
    with torch.no_grad():
        out = net(x, adj)
    assert out.shape == (3, 6)


def test_two_hop_net_consumes_edge_sign_and_weight():
    """Through BOTH hops, flipping an edge's sign OR changing its weight must change the net output."""
    torch.manual_seed(7)
    net = MaskedRichNet(hidden=8, heads=2, use_graph=True).eval()
    x = torch.randn(1, 5, 10, 5)
    A = torch.eye(5).unsqueeze(0).clone()
    A[0, 0, 1] = 0.6                                       # source 1 -> target 0
    with torch.no_grad():
        base = net(x, A)
        A_neg = A.clone(); A_neg[0, 0, 1] = -0.6
        out_neg = net(x, A_neg)
        A_w = A.clone(); A_w[0, 0, 1] = 0.1
        out_w = net(x, A_w)
    assert not torch.allclose(base[0, 0], out_neg[0, 0], atol=1e-6), "2-hop: sign flip ignored"
    assert not torch.allclose(base[0, 0], out_w[0, 0], atol=1e-6), "2-hop: weight change ignored"


def test_two_hop_mask_aware():
    """2-hop must stay mask-aware: with only self-loops, perturbing another node cannot change target
    0's output (no path reaches it through either hop)."""
    torch.manual_seed(8)
    net = MaskedRichNet(hidden=8, heads=2, use_graph=True).eval()
    x = torch.randn(1, 5, 10, 5)
    A = torch.eye(5).unsqueeze(0).clone()                 # self-loops only -> disconnected nodes
    with torch.no_grad():
        out1 = net(x, A)
        x2 = x.clone(); x2[0, 3] = torch.randn(10, 5)     # perturb an unconnected node
        out2 = net(x2, A)
    assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6), "2-hop leaked a non-neighbour into target"


def _synth_panel(tmp_path, n_days=400, tickers=("AAA", "BBB", "CCC")):
    """Write synthetic processed Parkinson CSVs + matching raw OHLCV; returns (files, price_dir)."""
    import pandas as pd
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    proc = tmp_path / "proc"; raw = tmp_path / "raw"; proc.mkdir(); raw.mkdir()
    for k, tk in enumerate(tickers):
        v = np.empty(n_days); v[0] = 1e-4 * (k + 1)
        for t in range(1, n_days):
            v[t] = 5e-5 * (k + 1) + 0.85 * v[t - 1] + 1e-5 * abs(rng.standard_normal())
        pd.DataFrame({"date": dates, "parkinson_volatility": v}).to_csv(proc / f"{tk}_processed.csv", index=False)
        close = 20.0 + np.cumsum(rng.normal(0, 0.2, n_days))
        span = np.sqrt(v) * close
        pd.DataFrame({"date": dates, "open": close, "high": close + span, "low": close - span,
                      "close": close, "volume": rng.integers(1e5, 1e6, n_days)}).to_csv(
            raw / f"{tk}_ohlcv.csv", index=False)
    return sorted(str(p) for p in proc.glob("*_processed.csv")), str(raw)


def test_train_only_invariance_no_leakage(tmp_path):
    """Perturbing the TEST-region rows of the inputs must NOT change any train-fitted quantity: the two
    train graph adjacencies and the per-node train target scaler are estimated on train rows only."""
    import pandas as pd
    files, price = _synth_panel(tmp_path)
    kw = dict(min_valid=2, min_train_rows=120)
    D0 = MR.build_masked_rich(files, price, lookback=10, horizon=1, **kw)
    # blow up the last 15% of dates (val+test region) in every processed + raw file
    for f in files:
        df = pd.read_csv(f); cut = int(len(df) * 0.85)
        df.loc[cut:, "parkinson_volatility"] *= 37.0
        df.to_csv(f, index=False)
        rf = Path(price) / (Path(f).name.replace("_processed.csv", "_ohlcv.csv"))
        rdf = pd.read_csv(rf); c2 = int(len(rdf) * 0.85)
        rdf.loc[c2:, ["high", "low", "volume"]] = rdf.loc[c2:, ["high", "low", "volume"]] * 37.0
        rdf.to_csv(rf, index=False)
    D1 = MR.build_masked_rich(files, price, lookback=10, horizon=1, **kw)
    assert D0.tickers == D1.tickers
    assert np.allclose(D0.adj_vol2pk, D1.adj_vol2pk), "vol->PK train edge changed after test-region perturbation"
    assert np.allclose(D0.adj_corr, D1.adj_corr), "corr train edge changed after test-region perturbation"
    assert np.allclose(D0.t_mean, D1.t_mean), "train target scaler changed after test-region perturbation"
    # the test targets themselves DID change (sanity: the perturbation reached the test fold)
    assert not np.allclose(D0.y_te[D0.tmask_te.astype(bool)], D1.y_te[D1.tmask_te.astype(bool)])


def test_run_out_subdir_writes_separate_results_tree(tmp_path, monkeypatch):
    """run(out_subdir=...) must write under results/<out_subdir>/, never clobbering the delivered
    masked_rich_floor1e2 tree (needed for the volatility-proxy robustness study)."""
    import run_masked_rich as RMR
    from config import SMOKE
    files, price = _synth_panel(tmp_path, tickers=tuple(f"T{i:02d}" for i in range(10)))
    monkeypatch.setattr(RMR, "REPO", tmp_path)          # redirect result writes into tmp
    res = RMR.run("synth", files, price, 1, SMOKE, with_corr=False, out_subdir="masked_rich_yz/parkinson")
    custom = tmp_path / "results" / "masked_rich_yz" / "parkinson" / "synth_h1" / "result.json"
    assert custom.exists()                              # wrote to the custom tree
    assert not (tmp_path / "results" / "masked_rich_floor1e2").exists()   # did NOT touch the delivered tree
    assert "metrics_per_seed" in res and "config" in res


def test_seed_metric_stats_mean_std_not_ensemble():
    """seed_metric_stats reports the MEAN (and std/min/max) of per-seed metrics -- not the metric of the
    seed-averaged ensemble -- so the paper's '5-seed mean' label is faithful."""
    from run_masked_rich import seed_metric_stats, _pred_dict
    y = np.array([[1.0, 2.0]])
    tmask = np.array([[True, True]])
    dates = np.array([0])
    d1 = _pred_dict(np.array([[1.0, 2.0]]), y, tmask, dates, 2)   # perfect -> mse 0
    d2 = _pred_dict(np.array([[3.0, 4.0]]), y, tmask, dates, 2)   # off by 2 -> mse 4
    st = seed_metric_stats([d1, d2], floor=1e-8)
    assert st["n"] == 2 and st["n_seeds"] == 2
    assert abs(st["mse"] - 2.0) < 1e-9            # mean of per-seed mse (0, 4)
    assert abs(st["mse_std"] - 2.0) < 1e-9        # std of (0, 4)
    assert st["mse_min"] == 0.0 and st["mse_max"] == 4.0
    assert st["per_seed"]["mse"] == [0.0, 4.0]


def test_ensemble_metric_differs_from_perseed_mean_schema_contract():
    """External review H-01: pin the schema contract that ``metrics`` (metric of the seed-AVERAGED
    ensemble, used only for the DM forecast) and ``metrics_per_seed`` (MEAN of seed-level metrics,
    the paper-reported number) are DISTINCT quantities -- so a future edit cannot silently conflate
    them. For the same two seeds: ensemble mse = 1.0, per-seed mean mse = 2.0."""
    from run_masked_rich import seed_metric_stats, _pred_dict, _ens, _metrics
    y = np.array([[1.0, 2.0]]); tmask = np.array([[True, True]]); dates = np.array([0])
    d1 = _pred_dict(np.array([[1.0, 2.0]]), y, tmask, dates, 2)   # perfect
    d2 = _pred_dict(np.array([[3.0, 4.0]]), y, tmask, dates, 2)   # off by 2
    ens_mse = _metrics(_ens([d1, d2]), 1e-8)["mse"]               # ens pred [2,3] -> mse 1.0
    ps_mean_mse = seed_metric_stats([d1, d2], 1e-8)["mse"]        # mean(0, 4) = 2.0
    assert abs(ens_mse - 1.0) < 1e-9
    assert abs(ps_mean_mse - 2.0) < 1e-9
    assert ens_mse != pytest.approx(ps_mean_mse)                  # the two fields must not be swapped


def test_har5_exposed_and_harx_ols_runs():
    """HAR-X fair baseline: build exposes the raw 5-feature vector at t, its first 3 cols are the HAR
    features, and a 5-feature linear OLS fits + yields finite predictions (isolates the extra-feature
    contribution so a deep/graph win over HAR is not just the 2 extra node features)."""
    D = MR.build_masked_rich(_VN30, _PRICE, lookback=10, horizon=5)
    assert D.har5_tr.shape[-1] == 5 and D.har5_te.shape[-1] == 5
    assert np.isfinite(D.har5_tr).all() and np.isfinite(D.har5_te).all()
    assert np.allclose(D.har5_tr[..., :3], D.har_tr)          # first 3 = daily/weekly/monthly HAR
    m = D.tmask_tr.astype(bool)
    Xb = np.column_stack([np.ones(int(m.sum())), D.har5_tr[m]])
    coef = np.linalg.lstsq(Xb, D.y_tr[m], rcond=None)[0]
    pred = np.column_stack([np.ones(len(D.har5_te.reshape(-1, 5))), D.har5_te.reshape(-1, 5)]) @ coef
    assert np.isfinite(pred).all()
