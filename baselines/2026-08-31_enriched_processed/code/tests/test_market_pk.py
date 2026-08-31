"""market_pk = cross-sectional MEAN of parkinson_variance over valid tickers on a sampled date + end-to-end build."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import enrich
from _synth import clean_frame, write_market


def test_compute_market_pk_equals_cross_sectional_mean():
    frames = {tk: enrich.build_ticker(clean_frame(n=40, seed=s))[0]
              for tk, s in [("A", 1), ("B", 2), ("C", 3)]}
    mpk = enrich.compute_market_pk(frames)
    sample = mpk.index[25]
    vals = [f.set_index("date")["parkinson_variance"].get(sample, np.nan) for f in frames.values()]
    vals = np.array([v for v in vals if np.isfinite(v)], float)
    assert abs(mpk.loc[sample] - vals.mean()) < 1e-12


def test_market_pk_ignores_nan_tickers():
    a = enrich.build_ticker(clean_frame(n=30, seed=1))[0]
    b = a.copy()
    b["parkinson_variance"] = np.nan             # a fully-invalid ticker must be excluded from the mean
    mpk = enrich.compute_market_pk({"A": a, "B": b})
    sample = mpk.index[20]
    assert abs(mpk.loc[sample] - a.set_index("date")["parkinson_variance"].loc[sample]) < 1e-12


def test_build_market_end_to_end_writes_and_fills_market_pk(tmp_path):
    price = write_market(tmp_path / "raw", {"XA": clean_frame(n=40, seed=1),
                                            "XB": clean_frame(n=40, seed=2)})
    out_root = tmp_path / "out"
    summary = enrich.build_market("mkt", price_dir=price, out_root=out_root, write=True)

    xa = pd.read_csv(out_root / "mkt" / "XA.csv")
    xb = pd.read_csv(out_root / "mkt" / "XB.csv")
    # market_pk on a sampled date equals the recomputed cross-sectional mean
    d = xa["date"].iloc[25]
    va = xa.set_index("date")["parkinson_variance"].loc[d]
    vb = xb.set_index("date")["parkinson_variance"].loc[d]
    got = xa.set_index("date")["market_pk"].loc[d]
    assert abs(got - np.nanmean([va, vb])) < 1e-12
    assert list(xa.columns) == enrich.ENRICHED_COLUMNS
    # schema-version sidecar written
    meta = json.loads((out_root / "mkt" / "_schema_version.json").read_text())
    assert meta["schema_version"] == enrich.SCHEMA_VERSION
    assert summary["n_tickers"] == 2


def test_build_market_attaches_regression_when_dir_given(tmp_path):
    price = write_market(tmp_path / "raw", {"XA": clean_frame(n=30, seed=1)})
    # a matching processed file so regression compares something
    xa_out = enrich.build_ticker(clean_frame(n=30, seed=1))[0]
    pdir = tmp_path / "proc"
    pdir.mkdir()
    xa_out[["date", "parkinson_variance"]].to_csv(pdir / "XA_processed.csv", index=False)
    summary = enrich.build_market("mkt", price_dir=price, out_root=tmp_path / "out",
                                  write=False, regression_dir=pdir)
    assert "regression" in summary
    assert summary["regression"]["worst_noncapped_diff"] < 1e-12


def test_build_market_writes_rejection_manifest(tmp_path):
    from _synth import dirty_frame
    price = write_market(tmp_path / "raw", {"DIRTY": dirty_frame()})
    out_root = tmp_path / "out"
    enrich.build_market("mkt", price_dir=price, out_root=out_root, write=True)
    rej = pd.read_csv(out_root / "mkt" / "DIRTY_rejections.csv")
    assert {"date", "reason"} == set(rej.columns)
    assert len(rej) >= 1


def test_build_market_skips_all_dropped_ticker_but_keeps_its_manifest(tmp_path):
    good = clean_frame(n=30, seed=1)
    junk = clean_frame(n=8, seed=2).copy()
    junk["high"] = np.nan                        # every bar naninf -> 0-row enriched frame
    price = write_market(tmp_path / "raw", {"GOOD": good, "JUNK": junk})
    out_root = tmp_path / "out"
    summary = enrich.build_market("mkt", price_dir=price, out_root=out_root, write=True)
    assert summary["n_tickers"] == 1             # JUNK excluded from the panel
    assert summary["n_empty_tickers"] == 1
    assert (out_root / "mkt" / "GOOD.csv").exists()
    assert not (out_root / "mkt" / "JUNK.csv").exists()          # no header-only CSV written
    assert (out_root / "mkt" / "JUNK_rejections.csv").exists()   # audit trail kept
    assert summary["rows_in"] >= 38              # honest raw count includes the dropped 8 rows


def test_build_market_no_write_and_limit(tmp_path):
    price = write_market(tmp_path / "raw", {"XA": clean_frame(n=30, seed=1),
                                            "XB": clean_frame(n=30, seed=2)})
    summary = enrich.build_market("mkt", price_dir=price, out_root=tmp_path / "out",
                                  write=False, limit=1)
    assert summary["n_tickers"] == 1
    assert not (tmp_path / "out").exists()       # write=False -> nothing on disk


def test_summarize_market_handles_all_nan_frames():
    # cover the est_n==0 and market_pk-empty branches
    a = enrich.build_ticker(clean_frame(n=25, seed=1))[0]
    for c in ("parkinson_variance", "garman_klass_variance",
              "rogers_satchell_variance", "yang_zhang_n20"):
        a[c] = np.nan
    empty_mpk = pd.Series(dtype=float)
    s = enrich.summarize_market("mkt", {"A": a}, {"A": pd.DataFrame(columns=["date", "reason"])},
                                {k: 0 for k in enrich.DIRTY_CLASSES}, 25, empty_mpk)
    assert np.isnan(s["estimator_mean"]["parkinson_variance"])
    assert s["market_pk"]["n_days"] == 0
    assert np.isnan(s["market_pk"]["mean"])
