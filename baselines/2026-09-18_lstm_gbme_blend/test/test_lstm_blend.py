"""Tests for the LSTM-feature / GBME-blend baseline: sequence construction (shape/left-pad/causality), the
real torch LSTM trainer (early-stop, learning curves), leakage-safe out-of-fold cross-fitting (coverage +
causality + fail-loud gap), the val-fit blend weight, run-structure smoke with the gate-required over/under-fit
evidence keys, verdict/success/DM logic, error-correlation, spike mask, and atomic checkpoint.

Fixtures use small synthetic panels + a cheap fake LSTM trainer so the walk-forward driver is exercised
end-to-end without real training; one separate test runs the real torch LSTM on tiny data."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import lstm_blend_config as C
import lstm_feat as L
import overfit_check as OF
import run_lstm_blend as R
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
        "daily_return": rng.normal(0, 0.01, n)})
    return S1._feat(base)


def _frames(n_tickers=3):
    return {f"TK{i}": _ticker_frame(seed=i).assign(ticker=f"TK{i}", sector=0) for i in range(n_tickers)}


def _fake_loader(market):
    return _frames(), {f"TK{i}": 0 for i in range(3)}, {}


def _fake_trainer(Xtr, ytr, Xva, yva, Xpred, seed, max_epochs, patience, record=False):
    """Cheap stand-in for the LSTM trainer: a deterministic per-row prediction (mean of the last timestep, so
    it varies by row -> the tree can split and GBME+lstmfeat != GBME) + a one-point learning curve when asked.
    No torch training -> the walk-forward driver runs fast in tests."""
    p = Xpred[:, -1, :].mean(dim=1).cpu().numpy().astype(np.float64)
    return p, ([{"epoch": 0, "train_mse": 1.0, "val_mse": 1.0}] if record else [])


def _tiny(monkeypatch):
    monkeypatch.setattr(C, "MIN_ROWS", {"sp500": 10, "default": 10})


# --------------------------------------------------------------------------- sequences
def test_build_sequences_shape_leftpad_and_causal():
    frames = _frames(2)
    a = R.FM.panel(frames, {}, 1)
    X, lv = L.build_sequences(a, R.OWN, C.SEQ_LEN)
    assert X.shape == (len(a), C.SEQ_LEN, len(R.OWN)) and lv.shape == (len(a),)
    # left-pad: a ticker's FIRST row repeats its own values across the whole window (rows 0..SEQ-1 identical)
    idx0 = a.index[a["ticker"] == "TK0"][0]
    assert np.allclose(X[idx0], X[idx0][0])
    # causal: the sequence's LAST step equals that row's own feature vector
    assert np.allclose(X[idx0][-1], a.loc[idx0, R.OWN].to_numpy(float))
    # target = log h-ahead variance
    assert np.allclose(lv, np.log(np.maximum(a["y"].to_numpy(float), R.FL)))


def test_inner_blocks_partition():
    dates = pd.bdate_range("2021-01-01", periods=9).to_numpy()
    blocks = L.inner_blocks(dates, 3)
    assert len(blocks) == 3 and sum(len(b) for b in blocks) == 9
    assert set(np.concatenate(blocks)) == set(dates)


def test_val_split_trailing_dates():
    dates = pd.bdate_range("2021-01-01", periods=10).to_numpy()
    pos = np.arange(10)
    tr, va = L._val_split(dates, pos, 3)
    assert set(va) == {7, 8, 9} and set(tr) == set(range(7))     # trailing 3 dates held out


def test_to_variance_clips_both_sides():
    v = L._to_variance(np.array([np.log(R.FL) - 100.0, 0.0, np.log(C.PRED_CAP) + 100.0]))
    assert v[0] == pytest.approx(R.FL) and v[2] == pytest.approx(C.PRED_CAP)
    assert R.FL <= v[1] <= C.PRED_CAP


# --------------------------------------------------------------------------- real torch trainer
def _toy_tensors(n=64, f=4, seq=5, seed=0, noise=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, seq, f)).astype(np.float32)
    y = (X[:, -1, 0] if not noise else rng.normal(0, 1, n)).astype(np.float32)
    dev = L.DEVICE
    return (torch.as_tensor(X, device=dev), torch.as_tensor(y, device=dev))


def test_forward_batched_chunks_and_single():
    Xtr, ytr = _toy_tensors(n=5, seed=8)
    model = L.SeqLSTM(Xtr.shape[2], C.HIDDEN, C.LAYERS, C.DROPOUT).to(L.DEVICE)
    multi = L._forward_batched(model, Xtr, 2)                     # 5 rows / batch 2 -> 3 chunks -> torch.cat
    single = L._forward_batched(model, Xtr, 99)                   # one chunk -> returns outs[0]
    assert multi.shape == (5,) and single.shape == (5,)
    assert torch.allclose(multi, single, atol=1e-5)


def test_train_lstm_learns_and_records_curves():
    Xtr, ytr = _toy_tensors(seed=1)
    Xva, yva = _toy_tensors(seed=2)
    model, curves = L.train_lstm(Xtr, ytr, Xva, yva, seed=0, max_epochs=4, patience=3, record=True)
    assert isinstance(model, L.SeqLSTM)
    assert curves and set(curves[0]) == {"epoch", "train_mse", "val_mse"}
    model2, curves2 = L.train_lstm(Xtr, ytr, Xva, yva, seed=0, max_epochs=2, patience=3, record=False)
    assert curves2 == []                                          # record=False -> no curves


def test_train_lstm_early_stop_break(monkeypatch):
    """Noise target -> val MSE stops improving -> the non-improving `else` + early-stop `break` fire."""
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    Xtr, ytr = _toy_tensors(seed=3, noise=True)
    Xva, yva = _toy_tensors(seed=4, noise=True)
    # patience=2 so a non-improving epoch first loops back (bad=1 < patience: the 138->118 arc) before a second
    # non-improving epoch fires the early-stop break -> both branches of the early-stop guard are exercised.
    model, _ = L.train_lstm(Xtr, ytr, Xva, yva, seed=0, max_epochs=50, patience=2, record=False)
    assert isinstance(model, L.SeqLSTM)


def test_torch_trainer_predicts_and_curves():
    Xtr, ytr = _toy_tensors(seed=5)
    Xva, yva = _toy_tensors(seed=6)
    Xpred, _ = _toy_tensors(n=10, seed=7)
    p, curves = L.torch_trainer(Xtr, ytr, Xva, yva, Xpred, 0, 3, 2, record=True)
    assert p.shape == (10,) and np.isfinite(p).all() and curves
    p2, curves2 = L.torch_trainer(Xtr, ytr, Xva, yva, Xpred, 0, 2, 2, record=False)
    assert p2.shape == (10,) and curves2 == []


# --------------------------------------------------------------------------- OOF cross-fitting
def _seq_fixture():
    frames = _frames(2)
    a = R.FM.panel(frames, {}, 1)
    X, lv = L.build_sequences(a, R.OWN, C.SEQ_LEN)
    dates = a["date"].to_numpy()
    return a, X, lv, dates


def test_group_pred_raises_on_degenerate_val():
    a, X, lv, dates = _seq_fixture()
    single = np.where(dates == np.sort(np.unique(dates))[0])[0]   # one date -> no room for a val slice
    with pytest.raises(ValueError, match="degenerate"):
        L._group_pred(X, lv, dates, single, single, (0,), 2, 1, _fake_trainer)


def test_group_pred_seed_ensemble_averages():
    """Multi-seed path: two seeds -> the seed loop iterates twice and averages predictions (record only on the
    first). A constant fake trainer makes the seed-mean exactly that constant."""
    a, X, lv, dates = _seq_fixture()
    ts = pd.Timestamp(S1.FOLDS[0])
    pos_train = np.where((a.date >= S1.TRAIN_START) & (a.date < ts))[0]
    pos_pred = pos_train[:5]

    def const_trainer(Xtr, ytr, Xva, yva, Xpred, seed, me, pat, record=False):
        return np.full(len(Xpred), 0.5), ([{"epoch": 0, "train_mse": 1.0, "val_mse": 1.0}] if record else [])

    pred, curves = L._group_pred(X, lv, dates, pos_train, pos_pred, (0, 1), 2, 1, const_trainer, record=True)
    ymu = float(lv[pos_train].mean()); ysd = float(lv[pos_train].std()) + L.FL
    assert np.allclose(pred, 0.5 * ysd + ymu) and len(curves) == 1   # averaged; curves recorded once (seed 0)


def test_oof_coverage_and_causality(monkeypatch):
    """Every train row is predicted by an LSTM whose inner-train dates EXCLUDE its date; all rows filled."""
    a, X, lv, dates = _seq_fixture()
    ts = pd.Timestamp(S1.FOLDS[0])
    pos_train = np.where((a.date >= S1.TRAIN_START) & (a.date < ts))[0]
    calls = []

    def rec(Xa, la, da, inner_train_pos, held_pos, seeds, me, pat, trainer, record=False):
        calls.append((set(da[inner_train_pos]), set(da[held_pos])))
        return la[held_pos], []                                   # identity-ish; length matches held_pos

    monkeypatch.setattr(L, "_group_pred", rec)
    feat = L.oof_train_feat(X, lv, dates, pos_train, (0,), 2, 1, 123, k=3)
    all_d = set(dates[pos_train])
    covered = set()
    for tr_d, held_d in calls:
        assert tr_d.isdisjoint(held_d)                            # no row predicted by an LSTM that saw its date
        covered |= held_d
    assert covered == all_d and not np.isnan(feat).any() and feat.shape == (len(pos_train),)


def test_oof_raises_on_coverage_gap(monkeypatch):
    a, X, lv, dates = _seq_fixture()
    ts = pd.Timestamp(S1.FOLDS[0])
    pos_train = np.where((a.date >= S1.TRAIN_START) & (a.date < ts))[0]
    monkeypatch.setattr(L, "inner_blocks", lambda d, k: [np.sort(np.unique(d))[:3]])   # drops most dates
    with pytest.raises(RuntimeError, match="coverage"):
        L.oof_train_feat(X, lv, dates, pos_train, (0,), 2, 1, 0, k=3, trainer=_fake_trainer)


def test_test_feat_returns_curves():
    a, X, lv, dates = _seq_fixture()
    ts = pd.Timestamp(S1.FOLDS[0])
    pos_train = np.where((a.date >= S1.TRAIN_START) & (a.date < ts))[0]
    pos_test = np.where((a.date >= ts) & (a.date < pd.Timestamp(S1.FOLDS[1])))[0]
    feat, curves = L.test_feat(X, lv, dates, pos_train, pos_test, (0,), 2, 1, 0, trainer=_fake_trainer)
    assert feat.shape == (len(pos_test),) and (feat >= R.FL).all() and (feat <= C.PRED_CAP).all()
    assert curves and set(curves[0]) == {"epoch", "train_mse", "val_mse"}   # test_feat records curves


# --------------------------------------------------------------------------- pure logic
def test_best_blend_weight_prefers_better_source():
    y = np.full(50, 3e-4)
    gbme = np.full(50, 3e-4)                                      # perfect -> QLIKE minimal at w=1
    lstm = np.full(50, 3e-2)                                      # far off
    assert R._best_blend_weight(y, gbme, lstm) == 1.0


def test_err_corr_constant_and_normal():
    assert R._err_corr(np.ones(5), np.arange(5)) == 0.0          # constant -> undefined -> 0
    r = R._err_corr(np.arange(10.0), np.arange(10.0) * 2 + 1)
    assert r == pytest.approx(1.0)


def test_verdict_and_success():
    assert R.verdict(1.0, 0.01) is True
    assert R.verdict(-1.0, 0.01) is False and R.verdict(1.0, 0.20) is False and R.verdict(0.0, 0.01) is False
    good = {"verdict": {"beats": True}}
    assert R.success({1: good, 5: good}) is True
    assert R.success({1: good, 5: {"verdict": {"beats": False}}}) is False
    assert R.success({1: good}) is False                         # missing h5


def test_safe_dm_identical_and_valueerror(monkeypatch):
    e = np.array([0.1, 0.2, 0.3, 0.4])
    d = np.array(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"], dtype="datetime64[ns]")
    assert R._safe_dm(e, e.copy(), d, 1)["p_value"] == 1.0
    r2 = R._safe_dm(e, e + 0.05, d, 1)
    assert 0.0 <= r2["p_value"] <= 1.0
    monkeypatch.setattr(R.ST, "date_clustered_dm", lambda *a, **k: (_ for _ in ()).throw(ValueError("hln")))
    assert R._safe_dm(e, e + 1.0, d, 1)["p_value"] == 1.0


def test_spike_mask_real_windows():
    dates = np.array(["2020-03-15", "2021-05-01", "2022-06-01", "2025-04-15", "2019-01-01"],
                     dtype="datetime64[ns]")
    assert list(R._spike_mask(dates)) == [True, False, True, True, False]


def test_load_earn_branches(monkeypatch, tmp_path):
    given = {"AAA": np.array(["2022-01-01"], dtype="datetime64[ns]")}
    assert R._load_earn("sp500", given) is given
    monkeypatch.setattr(R, "REPO", tmp_path)
    assert R._load_earn("hose", given) is given                  # non-sp500, no file -> unchanged
    d = tmp_path / "results" / "gamma_gbm"
    d.mkdir(parents=True)
    pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"],
                  "earnings_date": pd.to_datetime(["2022-03-01", "2022-06-01", "2022-04-01"])}
                 ).to_parquet(d / "hose_earnings_combined.parquet")
    out = R._load_earn("hose", {})
    assert set(out) == {"AAA", "BBB"} and len(out["AAA"]) == 2


def test_metrics5_keys():
    y = np.array([1e-4, 2e-4, 3e-4]); p = np.array([1.1e-4, 1.9e-4, 3.2e-4])
    assert set(R._metrics5(y, p)) == {"mse", "rmse", "mae", "r2", "qlike"}


def test_own8_single_source():
    assert len(R.OWN) == 8 and "rq" not in R.OWN
    assert set(R.OWN) < set(R.FM.OWN)


def test_checkpoint_atomic(tmp_path):
    doc = {"h": 1, "metrics": {"GBME": {"qlike": 0.1}}}
    out = tmp_path / "lstm_blend_x_h1.json"
    R._checkpoint(doc, out)
    assert json.loads(out.read_text()) == doc
    assert not (tmp_path / (out.name + ".tmp")).exists()


# --------------------------------------------------------------------------- run structure
def test_run_sp500_structure_and_evidence(monkeypatch, tmp_path):
    """Full (non-smoke) multi-horizon walk-forward with a fake trainer: exercises the horizon loop, empty-fold
    `continue`, blend, and the gate-required evidence blocks; no per-fold/spike block for sp500."""
    _tiny(monkeypatch)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=False, trainer=_fake_trainer)
    assert set(docs) <= set(C.HORIZONS) and 1 in docs
    for h, doc in docs.items():
        assert (tmp_path / f"lstm_blend_sp500_h{h}.json").exists()
        for blk in ("metrics", "train_metrics", "val_metrics"):
            for m in R.ORDER:
                assert set(doc[blk][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
        assert doc["fit_diagnostics"][R.FEAT]["status"] in ("ok", "overfit", "underfit")
        assert doc["learning_curves"] and set(doc["dm"]) == {f"{R.FEAT}_vs_{R.BASE}", f"{R.BLEND}_vs_{R.BASE}"}
        assert "lstm_standalone" in doc and "err_correlation_vs_gbme" in doc["lstm_standalone"]
        assert "per_fold_qlike" not in doc and "spike_robustness" not in doc
        _ok, probs = OF.check_result_evidence(doc)
        assert all("missing" not in p for p in probs), probs


def test_run_hose_with_earnings_perfold_and_spike(monkeypatch, tmp_path):
    """HOSE smoke with a fake trainer + injected earnings (covers the has_earn feature path): carries per-fold
    QLIKE + spike robustness; non-overlapping spike window keeps all test rows."""
    _tiny(monkeypatch)
    frames = _frames(3)
    edates = {tk: np.array(["2022-08-15", "2022-11-15"], dtype="datetime64[ns]") for tk in frames}
    monkeypatch.setattr(R, "_load_earn", lambda market, ed: edates)
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("2099-01-01", "2099-12-31"),))
    docs = R.run("hose", load_fn=lambda m: (frames, {}, {}), out_dir=tmp_path, smoke=True, trainer=_fake_trainer)
    doc = docs[1]
    assert "per_fold_qlike" in doc and set(doc["per_fold_qlike"]) == set(R.ORDER)
    sr = doc["spike_robustness"]
    assert sr["n_spike_obs"] == 0 and sr["n_ex_spike_obs"] == doc["n"] and "beats_ex_spike" in sr
    assert doc["blend_weights"] and 0.0 <= doc["blend_weights"][0] <= 1.0


def test_run_hose_all_in_spike(monkeypatch, tmp_path):
    """When every test date is inside a spike window (keep.any() False), spike_robustness is omitted."""
    _tiny(monkeypatch)
    monkeypatch.setattr(R, "_load_earn", lambda market, ed: {})
    monkeypatch.setattr(C, "SPIKE_WINDOWS", (("2000-01-01", "2100-01-01"),))
    docs = R.run("hose", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=_fake_trainer)
    assert "spike_robustness" not in docs[1] and "per_fold_qlike" in docs[1]


def test_outer_fold_causality(monkeypatch, tmp_path):
    """Walk-forward causality: mutating rows AT/AFTER the fold's test end must not change the fold's result."""
    _tiny(monkeypatch)
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
    assert clean[1]["metrics"] == dirty[1]["metrics"]


@pytest.mark.smoke
def test_run_real_lstm_smoke(monkeypatch, tmp_path):
    """End-to-end happy path with the REAL torch LSTM (trainer=None): exercises train_lstm + torch_trainer +
    the OOF/test wiring inside the driver (the fake-trainer runs bypass all of that)."""
    _tiny(monkeypatch)
    monkeypatch.setattr(C, "EPOCHS_SMOKE", 3)
    monkeypatch.setattr(C, "INNER_K", 2)
    monkeypatch.setattr(C, "MIN_EPOCH", 0)
    docs = R.run("sp500", load_fn=_fake_loader, out_dir=tmp_path, smoke=True, trainer=None)
    doc = docs[1]
    for m in R.ORDER:
        assert set(doc["metrics"][m]) == {"mse", "rmse", "mae", "r2", "qlike"}
    assert doc["fit_diagnostics"][R.FEAT]["status"] in ("ok", "overfit", "underfit")
    assert doc["learning_curves"]["fold0"]                       # real per-epoch curves recorded
