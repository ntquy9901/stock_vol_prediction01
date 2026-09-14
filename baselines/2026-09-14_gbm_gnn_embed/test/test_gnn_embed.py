"""Tests for the GBME+GNN-embed baseline: embedding shape/weight-compat, OUT-OF-FOLD cross-fitting coverage
+ causality (no leakage), z->GBM plumbing signal-recovery + noise-neutrality, run-structure smoke with the
gate-required over/under-fit evidence keys, verdict/success logic, atomic checkpoint, spike mask.

Fixtures use small synthetic panels + a cheap fake GNN trainer so the walk-forward driver is exercised
end-to-end without real GNN training (the real trainer is covered separately by test_train_embedder_*)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import embed as E
import gnn_embed_config as C
import gnnhar_sp500 as G
import overfit_check as OF
import run_gnn_embed as R
import vn_gbm_graph_stage1 as S1


# --------------------------------------------------------------------------- synthetic data
def _ticker_frame(seed, start="2021-01-01", end="2023-06-30"):
    """One ticker's enriched frame: AR(1) Parkinson variance (so shift(-h) is learnable) + the enrichment
    columns S1._feat needs, passed through S1._feat to get logpk/rq/mr_* exactly like real load."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)
    pk = np.empty(n)
    pk[0] = 3e-4
    for t in range(1, n):
        pk[t] = max(1e-6, 0.85 * pk[t - 1] + 0.15 * 3e-4 + rng.normal(0, 3e-5))
    s = pd.Series(pk)
    base = pd.DataFrame({
        "date": dates, "parkinson_variance": pk,
        "har_daily": pk, "har_weekly": s.rolling(5, min_periods=1).mean().to_numpy(),
        "har_monthly": s.rolling(22, min_periods=1).mean().to_numpy(),
        "volume_zscore_22": rng.normal(0, 1, n), "market_pk": pk * 0.9,
        "daily_return": rng.normal(0, 0.01, n),
    })
    return S1._feat(base)


def _frames(n_tickers=2):
    return {f"TK{i}": _ticker_frame(seed=i).assign(ticker=f"TK{i}", sector=0) for i in range(n_tickers)}


def _fake_loader(market):
    return _frames(), {"TK0": 0, "TK1": 0}, {}


def _fake_trainer(X, Ys, Mt, adj, tr_idx, va_idx, in_f, seeds, max_epochs, patience, emb_idx, row_in, cidx,
                  record=False):
    """Cheap stand-in for the GNN embedder: deterministic per-row embeddings (varying by row/cell so the
    tree can split on them and GBME+z != GBME -> DM is well-defined), + a one-point learning curve when
    asked. No torch training -> the walk-forward driver runs fast in tests."""
    ri = np.asarray(row_in, float)[:, None]
    ci = np.asarray(cidx, float)[:, None]
    j = np.arange(G.N_HID)[None, :]
    z = (0.01 * np.sin(ri + ci + j)).astype(np.float32)
    return z, ([{"epoch": 0, "train_qlike": 1.0, "val_qlike": 1.0}] if record else [])


# --------------------------------------------------------------------------- embedding
def test_embed_shape_and_weight_compat():
    emb = E.GNNEmbedder(4, G.N_HID, C.N_GCN)
    x = torch.randn(2, 3, 4)
    adj = torch.eye(3)
    z = emb.embed(x, adj)
    assert z.shape == (2, 3, G.N_HID)
    # no extra params -> a trained GNNHAR's state loads into the embedder unchanged and vice versa
    plain = G.GNNHAR(4, G.N_HID, C.N_GCN)
    emb.load_state_dict(plain.state_dict())
    plain.load_state_dict(emb.state_dict())


def test_train_embedder_and_seed_embed_run():
    D, N, F = 40, 3, 4
    dev = E.DEVICE
    rng = np.random.default_rng(0)
    X = torch.as_tensor(rng.normal(0, 1, (D, N, F)).astype(np.float32), device=dev)
    Ys = torch.as_tensor(np.abs(rng.normal(1, 0.2, (D, N))).astype(np.float32), device=dev)
    Mt = torch.ones(D, N, device=dev)
    W = np.abs(rng.normal(0, 1, (N, N))); W /= W.sum(1, keepdims=True)
    adj = torch.as_tensor(W.astype(np.float32), device=dev)
    tr_idx, va_idx = np.arange(0, 30), np.arange(30, 40)
    model, bv, curves = E.train_embedder(X, Ys, Mt, adj, tr_idx, va_idx, F, 0, 3, 1)
    assert isinstance(model, E.GNNEmbedder) and np.isfinite(bv) and curves == []
    _, _, curves2 = E.train_embedder(X, Ys, Mt, adj, tr_idx, va_idx, F, 0, 2, 1, record=True)
    assert curves2 and set(curves2[0]) == {"epoch", "train_qlike", "val_qlike"}
    emb_idx = np.arange(30, 40)
    row_in = np.array([0, 1, 2]); cidx = np.array([0, 1, 2])
    z, cv = E.seed_embed(X, Ys, Mt, adj, tr_idx, va_idx, F, (0,), 3, 1, emb_idx, row_in, cidx, record=True)
    assert z.shape == (3, G.N_HID) and cv
    z2, cv2 = E.seed_embed(X, Ys, Mt, adj, tr_idx, va_idx, F, (0,), 2, 1, emb_idx, row_in, cidx, record=False)
    assert z2.shape == (3, G.N_HID) and cv2 == []             # record=False -> no curves


def test_train_embedder_early_stop_triggers(monkeypatch):
    """Noise target -> val QLIKE fluctuates -> the non-improving `else` + early-stop `break` fire."""
    D, N, F = 24, 3, 3
    dev = E.DEVICE
    rng = np.random.default_rng(1)
    X = torch.as_tensor(rng.normal(0, 1, (D, N, F)).astype(np.float32), device=dev)
    Ys = torch.as_tensor(np.abs(rng.normal(1, 0.5, (D, N))).astype(np.float32), device=dev)  # noise -> no monotone gain
    Mt = torch.ones(D, N, device=dev)
    adj = torch.eye(N, device=dev)
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    _, bv, _ = E.train_embedder(X, Ys, Mt, adj, np.arange(0, 18), np.arange(18, 24), F, 0, 100, 2)
    assert np.isfinite(bv)


def test_arch_not_gcn_raises(monkeypatch):
    monkeypatch.setattr(C, "ARCH", "gat")
    with pytest.raises(NotImplementedError):
        E.train_embedder(torch.zeros(2, 2, 2), torch.ones(2, 2), torch.ones(2, 2), torch.eye(2),
                         np.array([0]), np.array([1]), 2, 0, 1, 1)


# --------------------------------------------------------------------------- OOF cross-fitting
def test_inner_blocks_partition():
    dates = pd.bdate_range("2021-01-01", periods=9).to_numpy()
    blocks = E.inner_blocks(dates, 3)
    assert len(blocks) == 3
    assert sum(len(b) for b in blocks) == 9
    assert set(np.concatenate(blocks)) == set(dates)


def test_oof_coverage_and_causality(monkeypatch):
    """Every train row is embedded by a GNN whose training dates EXCLUDE its date; blocks partition the
    train dates; all rows filled (fail-loud coverage). Verified by recording the (train,emb) date sets."""
    dates = pd.bdate_range("2021-01-01", periods=9)
    rows = []
    for tk in ("TK0", "TK1"):
        for d in dates:
            rows.append({"date": d, "ticker": tk, "y": 1.0})
    trf = pd.DataFrame(rows)
    calls = []

    def rec_group(df, tickers, feats, train_dates, emb_dates, graph_seed, seeds, me, pat, trainer,
                  record=False):
        calls.append((set(pd.to_datetime(train_dates)), set(pd.to_datetime(emb_dates))))
        er = df[df["date"].isin(emb_dates)]
        return np.zeros((len(er), G.N_HID), np.float32), er.index.to_numpy(), []

    monkeypatch.setattr(E, "_group_z", rec_group)
    z = E.oof_train_z(trf, ("TK0", "TK1"), ["y"], (0,), 1, 1, 123, k=3)
    all_dates = set(pd.to_datetime(dates))
    assert len(calls) == 3
    covered = set()
    for train_d, emb_d in calls:
        assert train_d.isdisjoint(emb_d)                     # no row embedded by a GNN that saw it
        assert train_d == all_dates - emb_d                  # inner-train is exactly the complement
        covered |= emb_d
    assert covered == all_dates                              # every train date embedded once (coverage)
    assert not np.isnan(z).any() and z.shape == (len(trf), G.N_HID)


def test_oof_raises_on_coverage_gap(monkeypatch):
    trf = pd.DataFrame({"date": pd.bdate_range("2021-01-01", periods=6).tolist() * 1,
                        "ticker": "TK0", "y": 1.0})

    def empty_group(df, tickers, feats, train_dates, emb_dates, gs, seeds, me, pat, trainer, record=False):
        return np.zeros((0, G.N_HID), np.float32), np.array([], int), []      # fills nothing

    monkeypatch.setattr(E, "_group_z", empty_group)
    with pytest.raises(RuntimeError, match="coverage"):
        E.oof_train_z(trf, ("TK0",), ["y"], (0,), 1, 1, 0, k=2)


# --------------------------------------------------------------------------- z -> GBM plumbing
def _plumbing_frames(informative):
    """train/test frames where y is driven by an extra feature z; OWN cols are uninformative noise."""
    rng = np.random.default_rng(4)
    n = 900
    zc = rng.normal(0, 1, n)
    y = 3e-4 * np.exp(0.6 * zc)                                # y depends on z (informative case)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in R.OWN})
    df["y"] = y
    df["z0"] = zc if informative else rng.normal(0, 1, n)      # noise z in the neutral case
    tr, te = df.iloc[:600], df.iloc[600:]
    return tr, te


def _qlike(tr, te, cols):
    import full_matrix as FM
    import metrics as M
    p = np.mean([FM.gbm(tr, te, cols, s) for s in (0, 1)], 0)
    return float(np.mean(M.per_obs_qlike(te["y"].to_numpy(float), p, floor=R.FL)))


def test_z_plumbing_signal_recovery():
    tr, te = _plumbing_frames(informative=True)
    q_base = _qlike(tr, te, R.OWN)
    q_z = _qlike(tr, te, R.OWN + ["z0"])
    assert q_z < q_base                                       # informative z lowers QLIKE (feature reaches tree)


def test_z_plumbing_noise_neutrality():
    tr, te = _plumbing_frames(informative=False)
    q_base = _qlike(tr, te, R.OWN)
    q_z = _qlike(tr, te, R.OWN + ["z0"])
    assert q_z <= q_base * 1.10                               # pure-noise z does not materially help or hurt


# --------------------------------------------------------------------------- run structure
def _tiny_min_rows(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


def test_run_sp500_structure_and_evidence(monkeypatch, tmp_path):
    """Full (non-smoke) multi-horizon walk-forward on synthetic SP500-style data with a fake trainer:
    exercises the horizon loop, empty-fold `continue`, and the gate-required evidence blocks."""
    _tiny_min_rows(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False, trainer=_fake_trainer)
    assert set(docs) <= set(C.HORIZONS) and 1 in docs
    for h, doc in docs.items():
        assert (tmp_path / f"gnn_embed_sp500_h{h}.json").exists()
        for blk in ("metrics", "train_metrics", "val_metrics"):
            for m in (R.BASE, R.LEARNED):
                assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
        assert doc["fit_diagnostics"][R.LEARNED]["status"] in ("ok", "overfit", "underfit")
        assert doc["learning_curves"] and "verdict" in doc
        assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
        # gate reads the evidence: blocks must be PRESENT (schema-compliant). Fit status is data-dependent
        # (synthetic spurious z may register 'overfit'); assert no MISSING-block complaint, not overall ok.
        _ok, probs = OF.check_result_evidence(doc)
        assert all("missing" not in p for p in probs), probs


def test_run_hose_perfold_and_spike(monkeypatch, tmp_path):
    _tiny_min_rows(monkeypatch)
    monkeypatch.setattr(R, "_load_earn", lambda market, edates: {})       # deterministic: no earnings file
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("2099-01-01", "2099-12-31"),))  # non-overlap -> keep all
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=_fake_trainer)
    doc = docs[1]
    assert "per_fold_qlike" in doc and set(doc["per_fold_qlike"]) == {R.BASE, R.LEARNED}
    sr = doc["spike_robustness"]
    assert sr["n_spike_obs"] == 0 and sr["n_ex_spike_obs"] == doc["n"]
    assert "gain_pct_ex_spike" in sr and "beats_ex_spike" in sr


def test_outer_fold_causality(monkeypatch, tmp_path):
    """Walk-forward causality: mutating rows AT/AFTER the fold's test end must not change the fold's result
    (the fold only reads dates < ts-embargo for train and [ts,tend) for test)."""
    _tiny_min_rows(monkeypatch)
    frames = _frames()
    # true future = strictly beyond tend + the h-target reach (test labels y=pk.shift(-h) legitimately read up
    # to ~tend, so only dates past that margin are genuinely unused by fold 0).
    cutoff = pd.Timestamp(S1.FOLDS[1]) + pd.Timedelta(days=20)
    mutated = {}
    for tk, fr in frames.items():
        g = fr.copy()
        post = g["date"] >= cutoff
        for c in R.OWN + ["parkinson_variance", "logpk"]:      # corrupt genuine-future rows only
            g.loc[post, c] = g.loc[post, c] * 7.0 + 1.0
        mutated[tk] = g
    clean = R.run("sp500", load_fn=lambda m: (frames, {}, {}), out_dir=tmp_path / "a", smoke=True,
                  trainer=_fake_trainer)
    dirty = R.run("sp500", load_fn=lambda m: (mutated, {}, {}), out_dir=tmp_path / "b", smoke=True,
                  trainer=_fake_trainer)
    assert clean[1]["metrics"] == dirty[1]["metrics"]          # identical -> no future-fold leakage


@pytest.mark.smoke
def test_run_real_embedder_smoke(monkeypatch, tmp_path):
    """End-to-end happy path with the REAL GNN embedder (trainer=None): exercises the actual torch forward +
    embed() gather + OOF/test wiring inside the driver (the fake-trainer run tests bypass all of that)."""
    _tiny_min_rows(monkeypatch)
    monkeypatch.setattr(C, "EPOCHS_SMOKE", 5)
    monkeypatch.setattr(C, "INNER_K", 2)
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=None)
    doc = docs[1]
    for m in (R.BASE, R.LEARNED):
        assert set(doc["metrics"][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert doc["fit_diagnostics"][R.LEARNED]["status"] in ("ok", "overfit", "underfit")
    assert doc["learning_curves"]["fold0"]                     # real per-epoch curves recorded


def test_group_z_raises_on_single_train_date(monkeypatch):
    """_group_z fails loud when an inner-train has <2 dates (cannot hold out a validation date)."""
    dates = pd.to_datetime(["2021-01-04", "2021-01-05"])
    rows = [{"date": d, "ticker": tk, "y": 1.0, "logpk": -8.0, "f": 0.5}
            for d in dates for tk in ("TK0", "TK1")]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match=">=2"):
        E._group_z(df, ("TK0", "TK1"), ["f"], [dates[0]], [dates[1]], 0, (0,), 1, 1, _fake_trainer)


def test_run_hose_all_in_spike(monkeypatch, tmp_path):
    """When every test date is inside a spike window (keep.any() False), spike_robustness is omitted."""
    _tiny_min_rows(monkeypatch)
    monkeypatch.setattr(R, "_load_earn", lambda market, edates: {})
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("2000-01-01", "2100-01-01"),))   # covers all -> keep none
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=_fake_trainer)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]


# --------------------------------------------------------------------------- pure helpers
def test_verdict_and_success():
    assert R.verdict(1.0, 0.01) is True
    assert R.verdict(-1.0, 0.01) is False                     # negative gain
    assert R.verdict(1.0, 0.20) is False                      # p too high
    assert R.verdict(0.0, 0.01) is False                      # gain must be strictly positive
    good = {"verdict": {"beats": True}}
    assert R.success({1: good, 5: good}) is True
    assert R.success({1: good, 5: {"verdict": {"beats": False}}}) is False
    assert R.success({1: good}) is False                      # missing h5


def test_safe_dm_identical_losses():
    """Identical loss series (z ignored by the tree -> GBME+z == GBME) must not crash DM; returns p=1.0."""
    e = np.array([0.1, 0.2, 0.3, 0.4])
    dates = np.array(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"], dtype="datetime64[ns]")
    r = R._safe_dm(e, e.copy(), dates, 1)
    assert r["p_value"] == 1.0 and r["mean_diff"] == 0.0
    r2 = R._safe_dm(e, e + 0.05, dates, 1)                     # non-degenerate -> real DM path
    assert 0.0 <= r2["p_value"] <= 1.0
    r3 = R._safe_dm(e, e + np.array([0.01, -0.02, 0.03, -0.01]), dates, 9)  # h>=n_dates -> DM raises -> guard
    assert r3["p_value"] == 1.0


def test_load_earn_branches(monkeypatch, tmp_path):
    given = {"AAA": np.array(["2022-01-01"], dtype="datetime64[ns]")}
    assert R._load_earn("sp500", given) is given                # sp500 -> unchanged
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert R._load_earn("hose", given) is given                 # non-sp500, no file -> unchanged
    d = tmp_path / "results" / "gamma_gbm"
    d.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                  "earnings_date": pd.to_datetime(["2022-03-01", "2022-06-01", "2022-04-01"])}
                 ).to_parquet(d / "hose_earnings_combined.parquet")
    out = R._load_earn("hose", {})
    assert set(out) == {"AAA", "BBB"} and len(out["AAA"]) == 2   # parsed real crawled dates


def test_own8_single_source():
    own = R.OWN
    assert len(own) == 8 and "rq" not in own
    import full_matrix as FM
    assert set(own) < set(FM.OWN)


def test_with_z_columns():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    z = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    out = R._with_z(df, z, ["z0", "z1"])
    assert list(out["z0"]) == [1.0, 3.0] and list(out["z1"]) == [2.0, 4.0]
    assert "z0" not in df.columns                             # original untouched (copy)


def test_spike_mask_real_windows():
    dates = np.array(["2020-03-15", "2021-05-01", "2022-06-01", "2025-04-15", "2019-01-01"],
                     dtype="datetime64[ns]")
    mask = R._spike_mask(dates)
    assert list(mask) == [True, False, True, True, False]


def test_checkpoint_atomic(tmp_path):
    doc = {"h": 1, "metrics": {"GBME": {"qlike": 0.1}}}
    out = tmp_path / "gnn_embed_x_h1.json"
    R._checkpoint(doc, out)
    assert json.loads(out.read_text()) == doc
    assert not (tmp_path / (out.name + ".tmp")).exists()      # tmp replaced, not left behind
