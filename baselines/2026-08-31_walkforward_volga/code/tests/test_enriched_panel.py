"""Enriched reader: correct shapes/features/target, causal NaN handling, fail-loud coverage guards,
missing-column + keep-filter branches, and the fixed-split frozen universe."""
from __future__ import annotations

import numpy as np
import pytest

import pipeline_config as pc
import wf_enriched_panel as EP

LB, H = 8, 1


def test_feature_cols_sourced_from_config(vzcol):
    cols = EP._feature_cols()
    assert cols == ["parkinson_variance", "har_weekly", "har_monthly", "market_pk", vzcol]


def test_build_returns_5_features_target_shapes_and_causal_nan(synth_files):
    files, tickers = synth_files
    panel = EP.build_enriched_panel(files, LB, H, tickers)
    assert panel.N == len(tickers)
    assert panel.feats.shape[2] == pc.N_NODE_FEATURES == 5
    assert panel.pk.shape == panel.feats[:, :, 0].shape
    # feature 0 == parkinson_variance == pk (read directly, not recomputed)
    np.testing.assert_allclose(panel.feats[:, :, 0], panel.pk, equal_nan=True)
    # target-date alignment: target_dates[k] == dates[anchor_k + horizon]
    dn = panel.dates.to_numpy()
    np.testing.assert_array_equal(panel.target_dates, dn[panel.anchors + H])
    # causal handling: after impute, volume_zscore has NO NaN on a ticker's own dates
    own = ~np.isnan(panel.pk)
    assert not np.isnan(panel.feats[:, :, 4][own]).any()
    # anchors start no earlier than FIRST_VALID + lookback - 1 (monthly HAR leading NaN excluded)
    assert panel.anchors.min() >= pc.FIRST_VALID + LB - 1


def test_interior_volume_zscore_nan_imputed_to_zero(tmp_path, synth_writer, vzcol):
    tickers = [f"T{i:02d}" for i in range(10)]
    files = synth_writer(tmp_path, tickers, T=200, seed=3)
    # inject an INTERIOR volume_zscore NaN (a flat-volume window) into AAA
    import pandas as pd
    df = pd.read_csv(files[0], parse_dates=["date"])
    df.loc[120, vzcol] = np.nan
    df.to_csv(files[0], index=False)
    panel = EP.build_enriched_panel(files, LB, H, tickers)
    own = ~np.isnan(panel.pk[:, 0])
    assert not np.isnan(panel.feats[own, 0, 4]).any()   # imputed to 0.0 (neutral), not left NaN


def test_check_feature_coverage_raises_on_no_dates():
    feats = np.zeros((4, 2, 5))
    own = np.ones((4, 2), bool)
    own[:, 0] = False                                    # ticker 0 has no trading dates
    with pytest.raises(ValueError, match="no trading dates"):
        EP._check_feature_coverage(feats, own, ["AAA", "BBB"])


def test_check_feature_coverage_raises_on_all_nan_feature():
    feats = np.ones((4, 2, 5))
    feats[:, 0, 2] = np.nan                              # har_monthly all-NaN on ticker 0's own dates
    own = np.ones((4, 2), bool)
    with pytest.raises(ValueError, match="har_monthly"):
        EP._check_feature_coverage(feats, own, ["AAA", "BBB"])


def test_build_raises_on_missing_column(tmp_path, synth_writer):
    tickers = ["AAA", "BBB"]
    files = synth_writer(tmp_path, tickers, T=200, seed=4, drop_col="market_pk")
    with pytest.raises(ValueError, match="missing columns"):
        EP.build_enriched_panel(files, LB, H, tickers)


def test_build_drops_keep_ticker_without_file_and_needs_two(synth_files):
    files, tickers = synth_files
    panel = EP.build_enriched_panel(files, LB, H, tickers + ["ZZZ"])   # ZZZ has no file -> dropped
    assert "ZZZ" not in panel.tickers and panel.N == len(tickers)
    with pytest.raises(ValueError, match=r"<2"):
        EP.build_enriched_panel(files, LB, H, ["ZZZ", "YYY"])          # no matching files -> <2


def test_build_raises_when_no_anchor_has_enough_valid_nodes(tmp_path, synth_writer):
    tickers = ["AAA", "BBB", "CCC"]                                    # fewer than MIN_VALID_NODES
    files = synth_writer(tmp_path, tickers, T=200, seed=6)
    with pytest.raises(ValueError, match="valid nodes"):
        EP.build_enriched_panel(files, LB, H, tickers)


def test_frozen_universe_screen_and_rejection_filter(tmp_path, synth_writer):
    tickers = [f"T{i:02d}" for i in range(10)]
    files = synth_writer(tmp_path, tickers, T=360, seed=5)
    (tmp_path / "XXX_rejections.csv").write_text("date,foo\n2020-01-01,1\n", encoding="utf-8")
    files_with_rej = files + [str(tmp_path / "XXX_rejections.csv")]
    keep = EP.frozen_universe(files_with_rej, LB, H)                   # rejection file filtered out
    assert set(keep) == set(tickers)
    # a huge min-train-rows screen drops everyone -> raise
    with pytest.raises(ValueError, match="train-row screen"):
        EP.frozen_universe(files, LB, H, min_train_rows=10 ** 9)
