"""No-look-ahead test per causal column: perturbing rows > t must not change the value at t."""
from __future__ import annotations

import numpy as np

import enrich
from _synth import clean_frame

_CAUSAL = ["parkinson_variance", "garman_klass_variance", "rogers_satchell_variance", "yang_zhang_n20",
           "log_range", "daily_return", "har_daily", "har_weekly", "har_monthly",
           "volume_zscore_22", "volume_zscore_20"]


def test_future_rows_do_not_change_present_values():
    df = clean_frame(n=60, seed=7)
    out0, _, _ = enrich.build_ticker(df)

    p = 45                                    # perturb a later row (stay valid: no split, no drop)
    df2 = df.copy()
    df2.loc[p, "high"] = df2.loc[p, "high"] * 1.08
    df2.loc[p, "close"] = df2.loc[p, "close"] * 1.05   # < 50% -> no split-jump, bar stays valid
    df2.loc[p, "volume"] = df2.loc[p, "volume"] * 3.0
    out1, _, _ = enrich.build_ticker(df2)

    head0 = out0.iloc[:p]
    head1 = out1.iloc[:p]
    for col in _CAUSAL:
        a = head0[col].to_numpy(float)
        b = head1[col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"NaN pattern changed for {col}"
        assert np.allclose(a[both], b[both], atol=1e-12, rtol=0), f"look-ahead leak in {col}"


def test_perturbation_actually_changed_the_perturbed_row():
    # guard: if the perturbation were a no-op the look-ahead test would be vacuous
    df = clean_frame(n=60, seed=7)
    out0, _, _ = enrich.build_ticker(df)
    df2 = df.copy()
    df2.loc[45, "high"] = df2.loc[45, "high"] * 1.08
    out1, _, _ = enrich.build_ticker(df2)
    assert out0["parkinson_variance"].iloc[45] != out1["parkinson_variance"].iloc[45]
