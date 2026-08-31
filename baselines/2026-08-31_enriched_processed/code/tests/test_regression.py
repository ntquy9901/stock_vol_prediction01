"""Clean-bar regression: enriched parkinson_variance == delivered data/processed value on clean bars."""
from __future__ import annotations

import pandas as pd
import pytest

import enrich
from _synth import clean_frame


def _real_vn30(ticker):
    p = enrich.PRICE_DIRS["vn30"] / f"{ticker}_ohlcv.csv"
    if not p.exists():  # pragma: no cover - env guard
        pytest.skip(f"real vn30 raw missing: {p}")
    return enrich.build_ticker(pd.read_csv(p))[0]


def test_real_vn30_clean_bars_match_delivered_to_1e12():
    processed = enrich.REPO / "data" / "processed"
    if not processed.exists():  # pragma: no cover - env guard
        pytest.skip("data/processed missing")
    frames = {tk: _real_vn30(tk) for tk in ("ACB", "FPT")}
    reg = enrich.regression_vs_processed(frames, processed)
    assert reg["n_compared"] > 1000
    assert reg["worst_noncapped_diff"] < 1e-12


def test_regression_excludes_capped_and_dirty(tmp_path):
    # deterministic coverage of the capped + dirty + missing-file branches
    out = enrich.build_ticker(clean_frame(n=30, seed=1))[0]
    out.loc[0, "dirty_flag"] = True                       # dirty bar excluded from comparison
    proc = out[["date", "parkinson_variance"]].copy()
    proc.loc[1, "parkinson_variance"] = 0.1               # capped bar (>=0.1) -> excluded
    # a small (< cap) mismatch on a DIRTY bar -> excluded by dirty_flag, not by the cap
    proc.loc[2, "parkinson_variance"] = out.loc[2, "parkinson_variance"] + 0.001
    out.loc[2, "dirty_flag"] = True
    pdir = tmp_path / "proc"
    pdir.mkdir()
    proc.to_csv(pdir / "TK_processed.csv", index=False)
    reg = enrich.regression_vs_processed({"TK": out, "MISSING": out}, pdir)   # MISSING has no processed file
    assert reg["n_capped"] == 1
    assert reg["worst_noncapped_diff"] < 1e-12
    assert reg["n_compared"] >= 25


def test_regression_all_dirty_ticker_compares_nothing(tmp_path):
    # every bar dirty -> empty comparison set (covers the `if d.size:`-False path)
    out = enrich.build_ticker(clean_frame(n=20, seed=2))[0]
    out["dirty_flag"] = True
    proc = out[["date", "parkinson_variance"]].copy()
    pdir = tmp_path / "proc"
    pdir.mkdir()
    proc.to_csv(pdir / "AD_processed.csv", index=False)
    reg = enrich.regression_vs_processed({"AD": out}, pdir)
    assert reg["n_compared"] == 0
    assert reg["worst_noncapped_diff"] == 0.0
