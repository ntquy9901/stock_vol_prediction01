"""Tests for the leverage-augmentation runner (own vs own+semi_neg vs own+semi)."""
import json

import numpy as np
import pandas as pd

import config
import augment_leverage as A


def _frames(nt=25, n=195, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for t in range(nt):
        d = pd.DataFrame({"date": pd.bdate_range("2015-01-01", periods=n),
                          "daily_return": rng.standard_normal(n) * 0.01,
                          "parkinson_variance": np.abs(rng.standard_normal(n)) * 1e-4 + 1e-6})
        for c in ["har_daily", "har_weekly", "har_monthly", "garman_klass_variance",
                  "rogers_satchell_variance", "yang_zhang_n20", "rq", "mr_change", "mr_slope5",
                  "mr_slope10", "mr_dev5", "mr_z22"]:
            d[c] = np.abs(rng.standard_normal(n)) * 1e-4
        d["ticker"] = f"T{t}"; d["sector"] = t % 3
        out[f"T{t}"] = d
    return out


def test_feature_sets():
    s = A._sets()
    assert "rq" not in A.OWN and set(A.OWN) == set(A.FM.OWN) - {"rq"}   # rq dropped from baseline
    assert s["own"] == A.OWN
    assert s["own+semi_neg"] == A.OWN + ["semi_neg"]
    assert s["own+semi"] == A.OWN + ["semi_neg", "semi_pos"]


def test_own_set_single_source(monkeypatch):
    """config.own_set drops OWN_DROP and appends OWN_ADD (the one place to edit for add/drop)."""
    assert config.own_set(["a", "rq", "b"]) == ["a", "b"]              # default drops rq
    monkeypatch.setattr(config, "OWN_DROP", ("x",))
    monkeypatch.setattr(config, "OWN_ADD", ("semi_neg",))
    assert config.own_set(["a", "x", "b"]) == ["a", "b", "semi_neg"]   # drop x, append semi_neg


def test_run_smoke(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 500})
    monkeypatch.setattr(A.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    out = A.run("hose", load_fn=lambda m: (_frames(), {}, {}))
    r = out["h1"]
    assert r["has_earn"] is False
    assert set(r["qlike"]) == {"own", "own+semi_neg", "own+semi"}
    assert set(r["vs_own"]) == {"own+semi_neg", "own+semi"}          # baseline excluded from comparisons
    for m in ("own+semi_neg", "own+semi"):
        assert set(r["vs_own"][m]) == {"gain_vs_own_pct", "dm_p"}
        assert set(r["spike_robustness"][m]) == {"gain_vs_own_pct_ex_spike", "dm_p_ex_spike", "n_ex_spike"}
    assert set(r["decile_qlike"]) == {"own", "own+semi_neg"}
    assert set(r["decile_qlike"]["own"]) == set(range(A.N_DECILE))   # one QLIKE per realized-vol decile
    assert set(r["fit_diagnostics"]["own"]) == {"verdict", "train_qlike", "test_qlike"}
    json.dumps(out)


def test_run_with_earnings_uses_earn_base(monkeypatch):
    """When the market provides earnings dates, the champion baseline includes FM.EARN (SP500 path)."""
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 500, "default": 500})
    monkeypatch.setattr(A.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2015-08-01", "2100-01-01"])
    ed = {f"T{t}": pd.to_datetime(["2015-03-16", "2015-06-15"]).to_numpy() for t in range(25)}
    out = A.run("hose", load_fn=lambda m: (_frames(), {}, ed))
    r = out["h1"]
    assert r["has_earn"] is True
    assert set(r["qlike"]) == {"own", "own+semi_neg", "own+semi"}    # keys unchanged; cols now include EARN


def test_spike_and_decile_helpers():
    dates = pd.to_datetime(["2019-06-03", "2020-03-16", "2021-05-04", "2022-06-01"]).to_numpy()
    assert list(A._spike_mask(dates)) == [False, True, False, True]
    y = np.linspace(1e-5, 1e-2, 100)
    err = {"own": np.ones(100), "own+semi_neg": np.linspace(0, 2, 100)}
    dec = A._decile_qlike(y, err, n=A.N_DECILE)
    assert set(dec) == {"own", "own+semi_neg"} and set(dec["own"]) == set(range(A.N_DECILE))
    assert dec["own"][0] == 1.0                                       # own err is constant 1.0 per decile


def test_run_empty_when_no_fold(monkeypatch):
    monkeypatch.setattr(config, "HORIZONS", (1,))
    monkeypatch.setattr(config, "MIN_ROWS", {"sp500": 30000, "default": 10 ** 9})
    monkeypatch.setattr(A.S1, "FOLDS", ["2015-02-01", "2015-07-01", "2100-01-01"])
    assert A.run("hose", load_fn=lambda m: (_frames(), {}, {})) == {}
