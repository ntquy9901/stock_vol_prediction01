"""Tests for the GBME+DAE baseline: swap-noise corruption (fraction + same-column source), DAE forward/embed
shapes, burn-in standardiser, temporal val split, real DAE training + learning curves + early stop, frozen
single-basis coverage + causality (the DAE trains on burn-in only, never on test rows), z->GBM plumbing
signal-recovery + noise-neutrality, walk-forward run-structure smoke with the gate-required over/under-fit
evidence keys, verdict/success logic, DM degeneracy guard, atomic checkpoint, spike mask, earnings loader.

Fixtures use small synthetic panels + a cheap FAKE DAE trainer so the walk-forward driver is exercised
end-to-end without real torch training (the real trainer is covered separately by the train_dae/default_trainer
tests)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import dae as D
import dae_config as C
import overfit_check as OF
import run_dae as R
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


def _fake_loader_earn(market):
    ed = {tk: np.array(["2022-03-01", "2022-09-01"], dtype="datetime64[ns]") for tk in ("TK0", "TK1")}
    return _frames(), {"TK0": 0, "TK1": 0}, ed


def _fake_trainer(Xburn, burn_dates, Xall, k, epochs, patience, seed, record=False):
    """Cheap stand-in for the DAE embedder: deterministic per-row embeddings (varying by row/dim so the tree
    can split on them and GBME+DAE != GBME -> DM is well-defined) over ALL rows of Xall, + a one-point
    learning curve when asked. No torch training -> the walk-forward driver runs fast in tests."""
    n = len(Xall)
    ri = np.arange(n)[:, None]
    j = np.arange(k)[None, :]
    z = (0.01 * np.sin(ri + j)).astype(np.float32)
    return z, ([{"epoch": 0, "train_recon_mse": 1.0, "val_recon_mse": 1.0}] if record else [])


# --------------------------------------------------------------------------- swap noise
def test_swap_noise_fraction_and_same_column_source():
    """~rate of cells change; every changed cell's new value comes from the SAME column (another row)."""
    n, f, rate = 2000, 5, 0.3
    x = torch.tensor([[i * 10 + j for j in range(f)] for i in range(n)], dtype=torch.float32)
    gen = torch.Generator().manual_seed(0)
    xc = D.swap_noise(x, rate, gen)
    changed = xc != x
    frac = changed.float().mean().item()
    assert abs(frac - rate) < 0.02                                   # right fraction corrupted (expectation)
    for j in range(f):
        col_vals = set(x[:, j].tolist())
        new_vals = set(xc[changed[:, j], j].tolist())
        assert new_vals <= col_vals                                  # replacements drawn from the same column
    assert not torch.equal(xc, x)


def test_swap_noise_zero_rate_is_identity():
    x = torch.randn(50, 4)
    assert torch.equal(D.swap_noise(x, 0.0, torch.Generator().manual_seed(1)), x)


# --------------------------------------------------------------------------- DAE module
def test_dae_shapes_and_embed():
    m = D.DAE(6, (8, 4), C.K)
    x = torch.randn(10, 6)
    assert m.embed(x).shape == (10, C.K)
    assert m(x).shape == (10, 6)


def test_standardize_burnin_stats():
    Xburn = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]], np.float32)   # col1 constant -> std guarded to 1
    Xall = np.array([[3.0, 5.0], [7.0, 9.0]], np.float32)
    Xb, Xa = D.standardize(Xburn, Xall)
    assert np.allclose(Xb.mean(0), 0.0, atol=1e-6)
    assert np.allclose(Xa[:, 1], np.array([0.0, 4.0]))                   # constant col: (x-5)/1


def test_val_len_and_split():
    assert D.val_len(3) == 2 and D.val_len(100) == C.VALID_LEN and D.val_len(1) == 1
    dates = np.array(pd.bdate_range("2021-01-01", periods=5).tolist() * 2)
    tr, va = D.val_split(dates)
    assert len(tr) + len(va) == len(dates) and len(va) > 0


def test_train_dae_runs_records_and_early_stops(monkeypatch):
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (200, 4)).astype(np.float32)
    Xva = rng.normal(0, 1, (40, 4)).astype(np.float32)
    model, bv, curves = D.train_dae(Xtr, Xva, 4, (8,), C.K, 3, 1, 0)
    assert isinstance(model, D.DAE) and np.isfinite(bv) and curves == []
    _, _, curves2 = D.train_dae(Xtr, Xva, 4, (8,), C.K, 2, 1, 0, record=True)
    assert curves2 and set(curves2[0]) == {"epoch", "train_recon_mse", "val_recon_mse"}
    # Early-stop `break`: a train set the DAE fits fast (all rows identical) + an UNRELATED noise val set ->
    # val recon MSE stops improving after epoch 0, so with patience=1 & MIN_EPOCH=0 the `else`/`break` fires.
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    Xtr_flat = np.ones((200, 4), np.float32)
    _, bv3, _ = D.train_dae(Xtr_flat, Xva, 4, (8,), C.K, 100, 1, 1)
    assert np.isfinite(bv3)


def test_default_trainer_embeds_all_rows(monkeypatch):
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    rng = np.random.default_rng(2)
    burn = rng.normal(0, 1, (60, 4)).astype(np.float32)
    dates = np.repeat(pd.bdate_range("2021-01-01", periods=30).to_numpy(), 2)
    allrows = rng.normal(0, 1, (100, 4)).astype(np.float32)
    z, curves = D.default_trainer(burn, dates, allrows, C.K, 3, 1, 0, record=True)
    assert z.shape == (100, C.K) and curves


# --------------------------------------------------------------------------- frozen basis
def test_frozen_z_single_basis_covers_all_rows(monkeypatch):
    """Frozen variant trains ONE DAE on burn-in only and embeds EVERY row from it (one shared basis): a single
    trainer call whose burn rows come only from burn-in dates; the returned map covers all rows + curves."""
    dates = pd.bdate_range("2021-01-01", periods=9)
    rows = [{"date": d, "ticker": tk, "f0": float(i), "f1": -float(i)}
            for i, d in enumerate(dates) for tk in ("TK0", "TK1")]
    df = pd.DataFrame(rows)
    burnin = dates[:6]
    calls = []

    def rec_trainer(Xburn, burn_dates, Xall, k, epochs, patience, seed, record=False):
        calls.append((len(Xburn), len(Xall), set(pd.to_datetime(burn_dates))))
        return np.zeros((len(Xall), k), np.float32), (
            [{"epoch": 0, "train_recon_mse": 1.0, "val_recon_mse": 1.0}] if record else [])

    zmap, curves = D.frozen_z(df, ["f0", "f1"], burnin, C.K, 1, 1, 0, trainer=rec_trainer)
    assert len(calls) == 1                                             # ONE DAE -> one basis (no per-fold refit)
    n_burn, n_all, burn_set = calls[0]
    assert burn_set == set(pd.to_datetime(burnin))                     # DAE trained on burn-in dates only (causal)
    assert n_burn == len(df[df["date"].isin(burnin)])                  # burn rows = burn-in rows only, no test rows
    assert n_all == len(df)                                            # every row embedded by that one DAE
    assert set(zmap) == set(int(i) for i in df.index)                  # coverage: every row has an embedding
    assert next(iter(zmap.values())).shape == (C.K,) and curves        # curves recorded for the evidence


def test_frozen_z_raises_on_short_burnin():
    dates = pd.bdate_range("2021-01-01", periods=3)
    df = pd.DataFrame({"date": list(dates) * 2, "f0": 1.0})
    with pytest.raises(ValueError, match=">=2"):
        D.frozen_z(df, ["f0"], dates[:1], C.K, 1, 1, 0, trainer=_fake_trainer)


# --------------------------------------------------------------------------- z -> GBM plumbing
def _plumbing_frames(informative):
    rng = np.random.default_rng(4)
    n = 900
    zc = rng.normal(0, 1, n)
    y = 3e-4 * np.exp(0.6 * zc)                                        # y depends on z (informative case)
    df = pd.DataFrame({c: rng.normal(0, 1, n) for c in R.OWN})
    df["y"] = y
    df["z0"] = zc if informative else rng.normal(0, 1, n)             # noise z in the neutral case
    return df.iloc[:600], df.iloc[600:]


def _qlike(tr, te, cols):
    import metrics as M
    p = R._gbm_clip(tr, te, cols, (0, 1))
    return float(np.mean(M.per_obs_qlike(te["y"].to_numpy(float), p, floor=R.FL)))


def test_z_plumbing_signal_recovery():
    tr, te = _plumbing_frames(informative=True)
    assert _qlike(tr, te, R.OWN + ["z0"]) < _qlike(tr, te, R.OWN)      # informative z lowers QLIKE


def test_z_plumbing_noise_neutrality():
    tr, te = _plumbing_frames(informative=False)
    assert _qlike(tr, te, R.OWN + ["z0"]) <= _qlike(tr, te, R.OWN) * 1.10   # noise z ~neutral


def test_gbm_clip_bounds(monkeypatch):
    """The clip caps predictions at PRED_CAP and floors at FL (both compared models get the SAME bounds)."""
    tr, te = _plumbing_frames(informative=True)
    monkeypatch.setattr(C, "PRED_CAP", 3e-4)                           # force the upper clip to bind
    p = R._gbm_clip(tr, te, R.OWN, (0,))
    assert p.max() <= 3e-4 + 1e-12 and p.min() >= R.FL


# --------------------------------------------------------------------------- run structure
def _tiny_min_rows(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


def test_run_sp500_structure_and_evidence(monkeypatch, tmp_path):
    """Full (non-smoke) multi-horizon walk-forward on synthetic SP500-style data with a fake trainer:
    exercises the horizon loop, the frozen zmap fill-once + reuse across folds, and the evidence blocks."""
    _tiny_min_rows(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False, trainer=_fake_trainer)
    assert set(docs) <= set(C.HORIZONS) and 1 in docs
    for h, doc in docs.items():
        assert (tmp_path / f"dae_sp500_h{h}.json").exists()
        assert list(doc["learning_curves"]) == ["frozen"]             # ONE basis, not per-fold curves
        for blk in ("metrics", "train_metrics", "val_metrics"):
            for m in (R.BASE, R.LEARNED):
                assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
        assert doc["fit_diagnostics"][R.LEARNED]["status"] in ("ok", "overfit", "underfit")
        assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
        # AUTO-DETECTION path (the real pre-push gate calls this with NO explicit `learned=`): the DAE model
        # must be recognised as learned (else the gate silently skips the file). See code_review M-1 fix.
        assert OF.looks_learned(R.LEARNED)
        _ok, probs = OF.check_result_evidence(doc)
        assert all("missing" not in p for p in probs), probs


def test_run_hose_earn_perfold_and_spike(monkeypatch, tmp_path):
    """HOSE smoke with earnings present (cols_base includes EARN): per_fold + spike robustness blocks."""
    _tiny_min_rows(monkeypatch)
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("2099-01-01", "2099-12-31"),))   # non-overlap -> keep all
    docs = R.run("hose", load_fn=_fake_loader_earn, out_dir=tmp_path, smoke=True, trainer=_fake_trainer)
    doc = docs[1]
    assert "per_fold_qlike" in doc and set(doc["per_fold_qlike"]) == {R.BASE, R.LEARNED}
    sr = doc["spike_robustness"]
    assert sr["n_spike_obs"] == 0 and sr["n_ex_spike_obs"] == doc["n"]
    assert "gain_pct_ex_spike" in sr and "beats_ex_spike" in sr


def test_run_hose_all_in_spike(monkeypatch, tmp_path):
    """When every test date is inside a spike window (keep.any() False), spike_robustness is omitted."""
    _tiny_min_rows(monkeypatch)
    monkeypatch.setattr(R, "_load_earn", lambda market, edates: {})
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("2000-01-01", "2100-01-01"),))   # covers all -> keep none
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=_fake_trainer)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]


def test_outer_fold_causality(monkeypatch, tmp_path):
    """Walk-forward causality: mutating rows AT/AFTER the fold's test end must not change the fold's result
    (the fold only reads dates < ts-embargo for train and [ts,tend) for test)."""
    _tiny_min_rows(monkeypatch)
    frames = _frames()
    cutoff = pd.Timestamp(S1.FOLDS[1]) + pd.Timedelta(days=20)
    mutated = {}
    for tk, fr in frames.items():
        g = fr.copy()
        post = g["date"] >= cutoff
        for c in R.OWN + ["parkinson_variance", "logpk"]:
            g.loc[post, c] = g.loc[post, c] * 7.0 + 1.0
        mutated[tk] = g
    clean = R.run("sp500", load_fn=lambda m: (frames, {}, {}), out_dir=tmp_path / "a", smoke=True,
                  trainer=_fake_trainer)
    dirty = R.run("sp500", load_fn=lambda m: (mutated, {}, {}), out_dir=tmp_path / "b", smoke=True,
                  trainer=_fake_trainer)
    assert clean[1]["metrics"] == dirty[1]["metrics"]                  # identical -> no future-fold leakage


@pytest.mark.smoke
def test_run_real_dae_smoke(monkeypatch, tmp_path):
    """End-to-end happy path with the REAL torch DAE (trainer=None): exercises standardise -> swap-noise
    training -> embed inside the driver (the fake-trainer runs bypass all of that)."""
    _tiny_min_rows(monkeypatch)
    monkeypatch.setattr(C, "EPOCHS_SMOKE", 3)
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=None)
    doc = docs[1]
    for m in (R.BASE, R.LEARNED):
        assert set(doc["metrics"][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert doc["fit_diagnostics"][R.LEARNED]["status"] in ("ok", "overfit", "underfit")
    assert doc["learning_curves"]["frozen"]                            # real per-epoch DAE curves recorded


# --------------------------------------------------------------------------- pure helpers
def test_verdict_and_success():
    assert R.verdict(1.0, 0.01) is True
    assert R.verdict(-1.0, 0.01) is False
    assert R.verdict(1.0, 0.20) is False
    assert R.verdict(0.0, 0.01) is False
    good = {"verdict": {"beats": True}}
    assert R.success({1: good, 5: good}) is True
    assert R.success({1: good, 5: {"verdict": {"beats": False}}}) is False
    assert R.success({1: good}) is False


def test_safe_dm_degenerate_and_real():
    e = np.array([0.1, 0.2, 0.3, 0.4])
    dates = np.array(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"], dtype="datetime64[ns]")
    r = R._safe_dm(e, e.copy(), dates, 1)
    assert r["p_value"] == 1.0 and r["mean_diff"] == 0.0               # identical losses -> guard
    r2 = R._safe_dm(e, e + 0.05, dates, 1)
    assert 0.0 <= r2["p_value"] <= 1.0                                 # non-degenerate -> real DM
    r3 = R._safe_dm(e, e + np.array([0.01, -0.02, 0.03, -0.01]), dates, 9)   # h>=n_dates -> DM raises -> guard
    assert r3["p_value"] == 1.0


def test_load_earn_branches(monkeypatch, tmp_path):
    given = {"AAA": np.array(["2022-01-01"], dtype="datetime64[ns]")}
    assert R._load_earn("sp500", given) is given                      # sp500 -> unchanged
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert R._load_earn("hose", given) is given                       # non-sp500, no file -> unchanged
    d = tmp_path / "results" / "gamma_gbm"
    d.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                  "earnings_date": pd.to_datetime(["2022-03-01", "2022-06-01", "2022-04-01"])}
                 ).to_parquet(d / "hose_earnings_combined.parquet")
    out = R._load_earn("hose", {})
    assert set(out) == {"AAA", "BBB"} and len(out["AAA"]) == 2         # parsed real crawled dates


def test_own8_single_source():
    import full_matrix as FM
    assert len(R.OWN) == 8 and "rq" not in R.OWN and set(R.OWN) < set(FM.OWN)


def test_with_z_columns():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    z = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    out = R._with_z(df, z, ["z0", "z1"])
    assert list(out["z0"]) == [1.0, 3.0] and list(out["z1"]) == [2.0, 4.0]
    assert "z0" not in df.columns                                     # original untouched (copy)


def test_spike_mask_real_windows():
    dates = np.array(["2020-03-15", "2021-05-01", "2022-06-01", "2025-04-15", "2019-01-01"],
                     dtype="datetime64[ns]")
    assert list(R._spike_mask(dates)) == [True, False, True, True, False]


def test_checkpoint_atomic(tmp_path):
    doc = {"h": 1, "metrics": {"GBME": {"qlike": 0.1}}}
    out = tmp_path / "dae_x_h1.json"
    R._checkpoint(doc, out)
    assert json.loads(out.read_text()) == doc
    assert not (tmp_path / (out.name + ".tmp")).exists()              # tmp replaced, not left behind
