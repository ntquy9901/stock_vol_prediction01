"""Horizon plumbing: --horizon shifts the target by h; the OUT filename carries h{H}."""
from __future__ import annotations

import numpy as np

import run_volga_walkforward as WF
import wf_enriched_panel as EP
from wf_folds import make_folds

LB = 8


def test_default_out_path_encodes_horizon():
    assert str(WF.default_out_path(5)).endswith("walkforward_volga_vn100_h5.json")
    assert str(WF.default_out_path(1)).endswith("walkforward_volga_vn100_h1.json")


def test_market_glob_and_out_path():
    """--market wiring: glob + out path carry the market; MARKETS lists the 5 universes."""
    g = WF.enriched_glob("vn30")
    assert "processed_enriched" in g and "vn30" in g and g.endswith("*.csv")
    assert str(WF.default_out_path(10, "vn30")).endswith("walkforward_volga_vn30_h10.json")
    assert set(WF.MARKETS) >= {"vn30", "vn100", "hose", "hnx", "sp500"}


def test_horizon_shifts_target(synth_files):
    files, tickers = synth_files
    p1 = EP.build_enriched_panel(files, LB, 1, tickers)
    p5 = EP.build_enriched_panel(files, LB, 5, tickers)
    dn = p5.dates.to_numpy()
    np.testing.assert_array_equal(p5.target_dates, dn[p5.anchors + 5])
    # h5 loses 4 more trailing anchors than h1 (T - horizon upper bound)
    assert p5.anchors.max() < p1.anchors.max()
    # packed target y = pk[t + horizon]
    fold = make_folds(len(p5.anchors), int(len(p5.anchors) * 0.7), 40, 25, 5)[0]
    D = EP.pack_fold(p5, fold, LB, 5)
    tr_anchor = p5.anchors[fold.train]
    expected = np.nan_to_num(np.stack([p5.pk[t + 5] for t in tr_anchor]))
    np.testing.assert_allclose(D.y_tr, expected)
