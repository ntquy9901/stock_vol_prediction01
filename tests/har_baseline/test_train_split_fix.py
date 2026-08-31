"""
Regression tests for the HAR-baseline per-ticker temporal split fix.

Background (the bug this guards against):
    The original train_har_baseline() concatenated every ticker's FULL time
    series end-to-end (in arbitrary os.listdir order) and then applied a single
    global cut at 80% of the pooled array. Because tickers were concatenated
    whole rather than interleaved by date, that global cut put some tickers
    ENTIRELY in train and others ENTIRELY in test -- it was not a temporal split
    at all (violates CLAUDE.md Section 3.A).

    The fix computes a per-ticker chronological cut (80% of THAT ticker's own
    valid rows), so the pooled train set holds the chronologically-early portion
    of EVERY ticker and the pooled test set holds the chronologically-late
    portion of EVERY ticker -- matching HARVolatilityDataset's proven pattern.

These tests assert the fixed behavior directly:
    - every ticker contributes rows to BOTH train and test (old code fails this),
    - each ticker's test dates are strictly after its train dates (no leakage),
    - a real-data slice runs end-to-end and yields finite, plausible metrics.
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.har_baseline.train import load_har_train_test_split, train_har_baseline


def _write_synthetic_ticker(directory, ticker, n_rows, seed):
    """Write a synthetic <ticker>_processed.csv with distinct chronological dates."""
    rng = np.random.default_rng(seed)
    # Distinct, non-overlapping date ranges per ticker keep the leakage assertion
    # unambiguous; within a ticker the dates are strictly increasing.
    start = f"20{10 + seed:02d}-01-01"
    dates = pd.date_range(start=start, periods=n_rows, freq="D")
    vol = np.abs(rng.normal(0.02, 0.005, n_rows)) + 1e-4
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "parkinson_variance": vol})
    df.to_csv(os.path.join(directory, f"{ticker}_processed.csv"), index=False)


@pytest.fixture
def synthetic_data_dir(tmp_path):
    """Three fake tickers of DIFFERENT lengths -- the case that exposed the bug."""
    d = tmp_path / "processed"
    d.mkdir()
    _write_synthetic_ticker(d, "AAA", n_rows=60, seed=1)
    _write_synthetic_ticker(d, "BBB", n_rows=80, seed=2)
    _write_synthetic_ticker(d, "CCC", n_rows=120, seed=3)
    return str(d)


def test_every_ticker_contributes_to_both_splits(synthetic_data_dir):
    """The core regression: after the fix, no ticker is 100% in one split."""
    _, _, _, _, train_meta, test_meta = load_har_train_test_split(synthetic_data_dir)

    train_tickers = set(train_meta["ticker"].unique())
    test_tickers = set(test_meta["ticker"].unique())
    all_tickers = {"AAA", "BBB", "CCC"}

    assert train_tickers == all_tickers, (
        f"Every ticker must appear in train; missing "
        f"{all_tickers - train_tickers}"
    )
    assert test_tickers == all_tickers, (
        f"Every ticker must appear in test; missing "
        f"{all_tickers - test_tickers}"
    )


def test_per_ticker_chronological_no_leakage(synthetic_data_dir):
    """Each ticker's test dates come strictly after its own train dates."""
    _, _, _, _, train_meta, test_meta = load_har_train_test_split(synthetic_data_dir)

    for ticker in {"AAA", "BBB", "CCC"}:
        train_dates = pd.to_datetime(train_meta.loc[train_meta["ticker"] == ticker, "date"])
        test_dates = pd.to_datetime(test_meta.loc[test_meta["ticker"] == ticker, "date"])
        assert len(train_dates) > 0 and len(test_dates) > 0
        assert train_dates.max() < test_dates.min(), (
            f"{ticker}: train must be chronologically before test (leakage guard)"
        )


def test_split_ratio_is_per_ticker_eighty_twenty(synthetic_data_dir):
    """Each ticker is split near 80/20 of its OWN valid rows, not globally."""
    _, _, _, _, train_meta, test_meta = load_har_train_test_split(synthetic_data_dir)

    for ticker in {"AAA", "BBB", "CCC"}:
        n_train = (train_meta["ticker"] == ticker).sum()
        n_test = (test_meta["ticker"] == ticker).sum()
        n_total = n_train + n_test
        # int(0.8 * n_total) train rows expected.
        assert n_train == int(0.8 * n_total)
        assert n_test == n_total - int(0.8 * n_total)


@pytest.mark.smoke
def test_real_data_slice_runs_end_to_end(tmp_path):
    """Smoke: fixed pipeline runs on a small slice of real data with plausible metrics."""
    real_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "processed",
    )
    if not os.path.isdir(real_dir):
        pytest.skip("real data/processed not available")

    real_files = sorted(f for f in os.listdir(real_dir) if f.endswith("_processed.csv"))
    if len(real_files) < 3:
        pytest.skip("need at least 3 real processed CSVs for the smoke slice")

    slice_dir = tmp_path / "processed_slice"
    slice_dir.mkdir()
    for fname in real_files[:3]:
        pd.read_csv(os.path.join(real_dir, fname)).to_csv(slice_dir / fname, index=False)

    out_dir = tmp_path / "out"
    _model, metrics = train_har_baseline(str(slice_dir), output_dir=str(out_dir))

    # All metrics finite (assert_finite_metrics already enforces this inside).
    for value in metrics.values():
        assert np.isfinite(value)

    # RMSE in the ~1e-3 order of magnitude seen for volatility baselines here.
    assert 0.0 < metrics["rmse"] < 0.05
    assert metrics["qlike"] > 0.0
    assert os.path.exists(out_dir / "test_metrics.csv")
