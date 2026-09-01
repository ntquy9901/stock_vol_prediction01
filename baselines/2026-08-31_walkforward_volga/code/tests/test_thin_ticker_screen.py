"""Thin-history ticker screen (build_enriched_panel): a ticker whose har_weekly or har_monthly is
all-NaN (fewer valid days than the 5/22 rolling window) is an inadequate-history node -- it is dropped
and recorded, NOT raised on. Adequate-history tickers are kept and still guarded by the fail-loud
coverage check. Covers both operands of the drop condition (har_weekly-all-NaN and har_monthly-all-NaN).
"""
from __future__ import annotations

import wf_enriched_panel as EP


def test_thin_history_tickers_dropped(tmp_path, synth_writer):
    normal = [f"N{i:02d}" for i in range(10)]            # >= MIN_VALID_NODES adequate-history tickers
    synth_writer(tmp_path, normal, T=360, seed=3)
    synth_writer(tmp_path, ["THINW"], T=3, seed=4)       # T<5 -> har_weekly all-NaN (left operand)
    synth_writer(tmp_path, ["THINM"], T=10, seed=5)      # 5<=T<22 -> har_weekly ok, har_monthly all-NaN (right)
    files = [str(p) for p in tmp_path.glob("*.csv")]
    panel = EP.build_enriched_panel(files, lookback=8, horizon=1,
                                    keep_tickers=normal + ["THINW", "THINM"])
    assert "THINW" not in panel.tickers and "THINM" not in panel.tickers   # both thin tickers screened
    assert set(panel.tickers) == set(normal) and panel.N == 10             # all adequate ones kept


def test_no_thin_tickers_is_noop(tmp_path, synth_writer):
    """When every ticker has adequate history, none is dropped (the false branch of the drop guard)."""
    normal = [f"N{i:02d}" for i in range(10)]
    synth_writer(tmp_path, normal, T=360, seed=6)
    files = [str(p) for p in tmp_path.glob("*.csv")]
    panel = EP.build_enriched_panel(files, lookback=8, horizon=1, keep_tickers=normal)
    assert panel.N == 10
