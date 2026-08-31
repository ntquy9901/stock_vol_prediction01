"""ETL cleaning applied correctly: dirty bars flagged + fixed, drops recorded, flags set."""
from __future__ import annotations

import numpy as np

import enrich
from _synth import clean_frame, dirty_frame


def test_dirty_frame_cleaning_and_rejections():
    out, rej, counts = enrich.build_ticker(dirty_frame())
    reasons = set(rej["reason"])
    # drops recorded in the manifest (no silent deletion)
    assert "naninf" in reasons
    assert "leading_backfill" in reasons
    assert "nonpositive_unrecoverable" in reasons
    assert "high_lt_low_unrecoverable" in reasons
    # cleaning labels present on surviving bars
    applied = set(out["cleaning_applied"])
    assert "reconstruct_nonpositive" in applied
    assert "swap_high_low" in applied
    assert "widen_range" in applied
    assert "backadjust_split" in applied
    # every raw dirty class contributed at least once to the counts
    assert counts["nonpositive"] >= 1
    assert counts["high_lt_low"] >= 1
    assert counts["open_close_outside"] >= 1
    assert counts["zero_range"] >= 1
    assert counts["split_jump"] >= 1
    assert counts["naninf"] >= 1


def test_zero_range_and_zero_volume_flags_kept_not_deleted():
    out, _, _ = enrich.build_ticker(dirty_frame())
    assert bool(out["zero_range_flag"].any())
    assert bool(out["zero_volume_flag"].any())


def test_reconstructed_bar_yields_finite_positive_parkinson():
    out, _, _ = enrich.build_ticker(dirty_frame())
    # the reconstructed bar is a valid bar -> finite, non-negative parkinson
    rec = out.loc[out["cleaning_applied"] == "reconstruct_nonpositive", "parkinson_variance"]
    assert len(rec) >= 1
    assert np.isfinite(rec.to_numpy(float)).all()
    assert (rec.to_numpy(float) >= 0).all()


def test_clean_frame_has_no_dirty_no_rejections():
    out, rej, counts = enrich.build_ticker(clean_frame(n=40, seed=2))
    assert len(rej) == 0
    assert not bool(out["dirty_flag"].any())
    assert set(out["cleaning_applied"]) == {"none"}
    assert sum(counts[k] for k in enrich.DIRTY_CLASSES) == 0


def test_prepare_raw_drops_bad_dates_and_dedups():
    df = clean_frame(n=10, seed=8)
    df.loc[3, "date"] = "not-a-date"          # unparseable -> dropped
    df.loc[5, "date"] = df.loc[4, "date"]     # duplicate -> last kept
    out, _, _ = enrich.build_ticker(df)
    assert out["date"].is_unique
    assert len(out) == 8                       # 10 - 1 unparseable - 1 duplicate


def test_detect_dirty_without_volume_column():
    df = clean_frame(n=6, seed=9).drop(columns=["volume"])
    masks = enrich.detect_dirty(df)
    assert masks["dirty"].shape[0] == 6
    # absent volume -> no naninf from volume
    assert masks["_counts"]["naninf"] == 0


def test_volume_zscore_without_volume_is_all_nan():
    df = clean_frame(n=30, seed=10).drop(columns=["volume"])
    z = enrich._volume_zscore(df, 22)
    assert np.isnan(z).all()
