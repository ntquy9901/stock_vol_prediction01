"""Coverage tests for the SP500 volatility probes (full_matrix / vn_gbm_graph_stage1 / gnnhar_sp500).

Exercises the testable helpers of all three drivers on tiny synthetic frames + a small real-data slice
(a few tickers from data/processed_enriched) so the pre-push diff-cover gate reaches C0=100% / C1>=95%
on the changed lines. The argparse ``main()`` bodies and the full walk-forward training loops carry
``# pragma: no cover`` (true entry drivers); every other function is invoked here.

Behaviour is not altered by these tests — they only call the existing functions and assert shapes /
finiteness, so the probes' reported numbers are unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eda"))
import full_matrix as FM  # noqa: E402
import gnnhar_sp500 as G  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402

SP_DIR = REPO / "data" / "processed_enriched" / "sp500_clean"
VN_DIR = REPO / "data" / "processed_enriched" / "vn30"
DEVICE = G.DEVICE
SP_TK = ("AAPL", "MSFT", "ABT", "ADI")
VN_TK = ("ACB", "FPT", "VNM", "MWG", "HPG")


def _patch_glob(module, files):
    """Return a context-less swap of ``module.glob.glob`` -> constant file list (restored by caller)."""
    orig = module.glob.glob
    module.glob.glob = lambda pattern: files
    return orig


@pytest.fixture(scope="module")
def sp500_data():
    # a real _rejections path is skipped by load()'s endswith check BEFORE any read, so it need not exist
    files = [str(SP_DIR / f"{t}.csv") for t in SP_TK] + [str(SP_DIR / "ZZZ_rejections.csv")]
    orig = _patch_glob(FM, files)
    try:
        frames, sect, edates = FM.load("sp500")
    finally:
        FM.glob.glob = orig
    return frames, sect, edates


@pytest.fixture(scope="module")
def vn_fm_data():
    files = [str(VN_DIR / f"{t}.csv") for t in VN_TK] + [str(VN_DIR / "ZZZ_rejections.csv")]
    orig = _patch_glob(FM, files)
    try:
        frames, sect, edates = FM.load("vn30")   # else-branch: reads SECT csv, edates stays empty
    finally:
        FM.glob.glob = orig
    return frames, sect, edates


@pytest.fixture(scope="module")
def vn_s1_frames():
    files = [str(VN_DIR / f"{t}.csv") for t in VN_TK] + [str(VN_DIR / "ZZZ_rejections.csv")]
    orig = _patch_glob(S1, files)
    try:
        frames = S1.load("vn30")
    finally:
        S1.glob.glob = orig
    return frames


# ----------------------------------------------------------------- full_matrix.py
def test_fm_load_sp500(sp500_data):
    frames, sect, edates = sp500_data
    assert set(frames) == set(SP_TK)              # _rejections path skipped
    assert edates and "AAPL" in edates            # earnings parquet loaded (sp500 branch)
    assert "rq" in frames["AAPL"].columns and "parkinson_variance" in frames["AAPL"].columns


def test_fm_load_vn(vn_fm_data):
    frames, sect, edates = vn_fm_data
    assert set(frames) == set(VN_TK) and edates == {}   # else-branch, no earnings


def test_fm_signed_none():
    T = pd.to_datetime(["2020-01-01", "2020-01-02"]).to_numpy()
    nxt, prv = FM._signed(T, None)
    assert np.all(nxt == 1e9) and np.all(prv == 1e9)
    nxt2, prv2 = FM._signed(T, np.array([], dtype="datetime64[ns]"))
    assert np.all(nxt2 == 1e9)


def test_fm_panel_with_and_without_earn(sp500_data, vn_fm_data):
    sp_frames, _, edates = sp500_data
    a = FM.panel(sp_frames, edates, 1)
    assert "earn_prox" in a.columns and "y" in a.columns and len(a) > 0   # edates -> True branch, _signed real
    vn_frames = vn_fm_data[0]
    b = FM.panel(vn_frames, {}, 1)
    assert "earn_prox" not in b.columns and len(b) > 0                    # empty edates -> False branch


def test_fm_uniform_and_nb(sp500_data):
    sp_frames = sp500_data[0]
    a = FM.panel(sp_frames, {}, 1)
    tickers = sorted(a["ticker"].unique())
    n = len(tickers)
    W = FM._uniform(n)
    assert W.shape == (n, n) and np.allclose(np.diag(W), 0.0)
    fold = a[a["date"].isin(a["date"].unique()[:15])]
    res = FM.nb(fold, tickers, W)
    assert res.shape[0] == len(fold) and np.all(np.isfinite(res))


def test_fm_ols_and_gbm(sp500_data):
    sp_frames = sp500_data[0]
    a = FM.panel(sp_frames, {}, 1)
    days = np.sort(a["date"].unique())
    mid = days[len(days) // 2]
    tr = a[a["date"] < mid].sample(n=min(800, (a["date"] < mid).sum()), random_state=0)
    te = a[a["date"] >= mid].head(200)
    harq = FM._harq_ols(tr, te)
    har = FM._har_ols(tr, te)
    g = FM.gbm(tr, te, FM.OWN, 0)
    for p in (harq, har, g):
        assert p.shape[0] == len(te) and np.all(p >= FM.FL)


# ----------------------------------------------------------------- vn_gbm_graph_stage1.py
def test_s1_feat_and_load(vn_s1_frames):
    df = pd.read_csv(VN_DIR / "ACB.csv", parse_dates=["date"]).head(60)
    out = S1._feat(df)
    assert {"rq", "logpk", "mr_change", "mr_z22"} <= set(out.columns)
    assert set(vn_s1_frames) == set(VN_TK)
    assert S1.FL == 1e-8 and len(S1.FOLDS) == 9 and S1.TOPK == 10


def test_s1_panel_graph_and_gbm(vn_s1_frames):
    a = S1.panel(vn_s1_frames, 1)
    assert "sect_mean" in a.columns and "y" in a.columns
    tickers = sorted(a["ticker"].unique())
    n = len(tickers)
    days = np.sort(a["date"].unique())
    mid = days[len(days) // 2]
    train = a[a["date"] < mid]
    W, Wp = S1.build_graph(train, tickers, np.random.default_rng(0))
    assert W.shape == (n, n) and Wp.shape == (n, n)

    fold = a[a["date"].isin(days[:30])]
    gf = S1.graph_feats(fold, tickers, W, "")                    # dense W -> js.size>0 branch
    assert len(gf) == len(fold) and all(f in gf.columns for f in S1.GRAPH)
    W2 = W.copy()
    W2[-1] = 0.0
    gf2 = S1.graph_feats(fold, tickers, W2, "p")                 # empty last row -> js.size==0 branch
    assert "g_nb_volp" in gf2.columns

    tr = train.sample(n=min(800, len(train)), random_state=0)
    te = a[a["date"] >= mid].head(200)
    pred = S1._gbm(tr, te, S1.OWN)
    assert pred.shape[0] == len(te) and np.all(pred >= S1.FL)


def test_s1_std_cols():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})   # b constant -> replace(0,nan)->fillna(0)
    z, m = S1._std_cols(df)
    assert z.shape == (3, 2) and np.all(z[:, 1] == 0.0)


# ----------------------------------------------------------------- gnnhar_sp500.py
def test_graph_conv_layer_bias_and_nobias():
    x = torch.randn(2, 5, 3, device=DEVICE)
    adj = torch.eye(5, device=DEVICE)
    out_b = G.GraphConvLayer(3, 4, bias=True).to(DEVICE)(x, adj)     # bias not None branch
    out_n = G.GraphConvLayer(3, 4, bias=False).to(DEVICE)(x, adj)    # bias None branch
    assert out_b.shape == (2, 5, 4) and out_n.shape == (2, 5, 4)


@pytest.mark.parametrize("n_gcn", [1, 2])
def test_gnnhar_forward(n_gcn):
    model = G.GNNHAR(3, G.N_HID, n_gcn).to(DEVICE)
    x = torch.randn(4, 6, 3, device=DEVICE)
    adj = torch.eye(6, device=DEVICE)
    out = model(x, adj)
    assert out.shape == (4, 6) and torch.all(out >= 0)              # final ReLU


def test_qlike_loss():
    pred = torch.rand(3, 4, device=DEVICE) + 0.1
    y = torch.rand(3, 4, device=DEVICE) + 0.1
    mask = torch.ones(3, 4, device=DEVICE)
    loss = G._qlike_loss(pred, y, mask)
    assert torch.isfinite(loss)


def test_build_fold_tensors():
    dates = pd.to_datetime([f"2020-01-0{i}" for i in range(1, 7)])
    rows = []
    for i, d in enumerate(dates):
        for t in ("A", "B"):
            rows.append({"date": d, "ticker": t, "y": 0.001 * (i + 1),
                         "f_var": float(i), "f_const": 2.0})          # f_const -> sd<=1e-12 branch
    fold = pd.DataFrame(rows)
    cut = np.datetime64("2020-01-05")
    X, Y, Ys, mask, sc, dpos, cpos, dts = G.build_fold_tensors(
        fold, ["A", "B"], ["f_var", "f_const"], lambda d: d < cut)
    assert X.shape == (6, 2, 2) and mask.sum() == 12 and sc > 0
    assert np.allclose(X[:, :, 1], 0.0)                              # constant feature z-scored to 0


def _tiny_train_inputs(D=8, N=3, F=3):
    torch.manual_seed(0)
    X = torch.randn(D, N, F, device=DEVICE)
    Ys = torch.rand(D, N, device=DEVICE) + 0.5
    Mt = torch.ones(D, N, device=DEVICE)
    adj = torch.eye(N, device=DEVICE)
    return X, Ys, Mt, adj


def test_train_once_improve():
    X, Ys, Mt, adj = _tiny_train_inputs()
    tr = torch.as_tensor(np.array([0, 1, 2, 3, 4]), device=DEVICE)
    model, bv = G._train_once(X, Ys, Mt, adj, tr, np.array([5, 6]), 3, 2, 0, max_epochs=2, patience=1)
    assert model is not None and np.isfinite(bv)                     # best_state loaded (improve branch)


def test_train_once_break_and_no_best():
    X, Ys, Mt, adj = _tiny_train_inputs()
    Mt[np.array([6, 7])] = 0.0                                       # zero VAL mask -> vloss NaN every epoch
    tr = torch.as_tensor(np.array([0, 1, 2, 3, 4, 5]), device=DEVICE)
    model, bv = G._train_once(X, Ys, Mt, adj, tr, np.array([6, 7]), 3, 1, 1, max_epochs=25, patience=1)
    assert model is not None and not np.isfinite(bv)                 # never improved -> break at epoch>=20


def test_train_predict_normal(monkeypatch):
    X, Ys, Mt, adj = _tiny_train_inputs(D=6)
    m = G.GNNHAR(3, G.N_HID, 2).to(DEVICE)
    monkeypatch.setattr(G, "_train_once", lambda *a, **k: (m, 1.0))  # finite <1.6 -> no retry
    preds, bv = G.train_predict(X, Ys, Mt, adj, np.array([0, 1, 2]), np.array([3]),
                                np.array([4, 5]), 3, 2, 0, 2, 1)
    assert bv == 1.0 and preds.shape == (2, 3)


def test_train_predict_retry(monkeypatch):
    X, Ys, Mt, adj = _tiny_train_inputs(D=6)
    m = G.GNNHAR(3, G.N_HID, 2).to(DEVICE)
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return (m, 5.0)                                              # >1.6 -> drives the retry loop
    monkeypatch.setattr(G, "_train_once", fake)
    preds, bv = G.train_predict(X, Ys, Mt, adj, np.array([0, 1, 2]), np.array([3]),
                                np.array([4, 5]), 3, 2, 0, 2, 1)
    assert bv == 5.0 and calls["n"] == 5                             # 1 initial + 4 retries


def test_pool_write(tmp_path):
    base = ["HAR", "GBM", "GBM+corr"]
    gnn_names = ["GNNHAR2L-corr-HAR3", "GNNHAR2L-none-HAR3", "GNNHAR2L-plac-HAR3",
                 "GNNHAR1L-corr-HAR3", "GNNHAR2L-corr-OWN9", "GNNHAR2L-none-OWN9",
                 "GNNHAR2L-plac-OWN9"]
    n = 40
    dates = np.array([np.datetime64("2020-01-01") + np.timedelta64(i, "D") for i in range(n)])
    yy = [np.random.default_rng(99).uniform(0.5, 1.5, n)]
    preds = {name: [np.random.default_rng(i + 1).uniform(0.5, 1.5, n)]
             for i, name in enumerate(base + gnn_names)}
    gnn_seed_q = {name: ([0.10, 0.20] if i % 2 == 0 else []) for i, name in enumerate(gnn_names)}
    out = {}
    outpath = tmp_path / "g.json"
    G.pool_write(out, 1, (0,), base, gnn_names, preds, yy, [dates], gnn_seed_q, outpath, verbose=False)
    G.pool_write(out, 1, (0,), base, gnn_names, preds, yy, [dates], gnn_seed_q, outpath, verbose=True)
    assert "h1" in out and out["h1"]["n"] == n and outpath.exists()
    assert set(out["h1"]["qlike"]) == set(base + gnn_names)
