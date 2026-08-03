"""Tests for the normalization + split-first fixes in dataset.py.

Covers two bugs in MultiStockDataset / create_multi_stock_dataloaders (the
HAR-only comparison baseline's data pipeline behind train.py / train_parallel.py):

Bug 1 (leakage): create_multi_stock_dataloaders used to build three FULL
MultiStockDataset instances (each computing HAR + fitting normalizers over the
whole unsplit series) and then positionally Subset them. It now loads raw data,
splits it chronologically, generates HAR per split, and fits normalizers on the
TRAIN split only.

Bug 2 (dead normalization): MultiStockDataset.__getitem__ fitted normalizers but
never applied .transform(), so training/eval ran on raw ~1e-3 volatility values.
__getitem__ now actually normalizes per stock (mirroring dataset_presplit.py).

House style follows test_date_alignment_fix.py: pytest, synthetic small
fixtures, plus a real-data-sample smoke test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.data_normalization import VolatilityNormalizer
from src.lstm_gat_hybrid.dataset import (
    MultiStockDataset,
    create_multi_stock_dataloaders,
)

pytestmark = pytest.mark.smoke


def _make_har_df(vol):
    """Build a HAR-feature DataFrame (the shape _generate_har_for_split emits)."""
    vol = np.asarray(vol, dtype=float)
    return pd.DataFrame({
        "har_daily_vol": vol,
        "har_weekly_vol": vol * 1.1,
        "har_monthly_vol": vol * 0.9,
        "parkinson_volatility": vol,
    })


def _fit_train_style_normalizers(dataset):
    """Fit per-stock normalizers on the dataset's own sequences (as the train
    branch of create_multi_stock_dataloaders does)."""
    for stock_idx, stock_name in enumerate(dataset.stock_names):
        feats = np.concatenate(
            [seq[0][:, stock_idx, :] for seq in dataset.sequences], axis=0)
        targs = np.array([seq[2][stock_idx] for seq in dataset.sequences])
        fn = VolatilityNormalizer().fit(feats)
        tn = VolatilityNormalizer().fit(targs.reshape(-1, 1))
        dataset.feature_normalizers[stock_name] = fn
        dataset.target_normalizers[stock_name] = tn


class TestGetItemAppliesNormalization:
    def _build_dataset(self):
        rng = np.random.default_rng(0)
        n = 14
        har = {
            "AAA": _make_har_df(0.001 + 0.0005 * rng.random(n)),
            "BBB": _make_har_df(0.02 + 0.01 * rng.random(n)),  # very different scale
        }
        raw = {k: v.copy() for k, v in har.items()}
        return MultiStockDataset(
            seq_length=3,
            forecast_horizon=1,
            graph_method="correlation",
            normalize=True,
            data_augmentation=False,
            precomputed_raw_data=raw,
            precomputed_har_data=har,
        )

    def test_getitem_output_is_on_normalized_scale(self):
        ds = self._build_dataset()
        _fit_train_style_normalizers(ds)

        x_raw, _adj, y_raw, _g = ds.sequences[0]
        x, _adj_t, y, _g2 = ds[0]
        x = x.numpy()
        y = y.numpy()

        # Actually transformed, not passed through raw.
        assert not np.allclose(x, x_raw), "features must be normalized, not raw"
        assert not np.allclose(y, y_raw), "targets must be normalized, not raw"

        # Matches an explicit per-stock transform of the raw sequence.
        for s, stock in enumerate(ds.stock_names):
            fn = ds.feature_normalizers[stock]
            tn = ds.target_normalizers[stock]
            for f in range(x_raw.shape[2]):
                expected = fn.transform(x_raw[:, s, f:f + 1]).flatten()
                assert np.allclose(x[:, s, f], expected, atol=1e-5)
            expected_y = np.clip(
                tn.transform(y_raw[s:s + 1].reshape(1, -1)).flatten()[0], -10.0, 10.0)
            assert np.allclose(y[s], expected_y, atol=1e-5)

        # BBB's raw targets are ~0.02 (not normalized); normalized targets should
        # be O(1), nowhere near the raw ~1e-2 scale.
        assert np.abs(y).max() > 0.1, "normalized targets should be O(1), not ~1e-3"

    def test_missing_normalizer_falls_back_to_raw(self):
        ds = self._build_dataset()  # normalizers left empty
        x_raw, _adj, y_raw, _g = ds.sequences[0]
        x, _adj_t, y, _g2 = ds[0]
        # With no fitted normalizers, __getitem__ falls back to raw values.
        assert np.allclose(x.numpy(), x_raw, atol=1e-6)
        assert np.allclose(y.numpy(), y_raw, atol=1e-6)

    def test_normalized_target_is_clipped(self):
        ds = self._build_dataset()
        # Tiny std -> a normal-scale raw target maps to a huge z-score.
        for stock in ds.stock_names:
            fn = VolatilityNormalizer()
            fn.set_params({"mean": 0.0, "std": 1.0})
            tn = VolatilityNormalizer()
            tn.set_params({"mean": 0.0, "std": 1e-6})
            ds.feature_normalizers[stock] = fn
            ds.target_normalizers[stock] = tn
        _x, _adj, y, _g = ds[0]
        assert y.numpy().max() <= 10.0 + 1e-6
        assert y.numpy().min() >= -10.0 - 1e-6
        # Raw targets are positive ~1e-2, so with std=1e-6 they saturate at +10.
        assert np.allclose(y.numpy(), 10.0, atol=1e-4)


def _write_synthetic_csvs(tmp_path, n_rows=200, n_tickers=3):
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-01", periods=n_rows, freq="B").strftime("%Y-%m-%d")
    for t in range(n_tickers):
        # Distinct positive volatility level per ticker + mild random walk.
        base = 0.005 * (t + 1)
        vol = np.abs(base + 0.001 * np.cumsum(rng.normal(0, 1, n_rows)) * 0.01 + 1e-4)
        df = pd.DataFrame({"date": dates, "parkinson_volatility": vol})
        df.to_csv(tmp_path / f"TICK{t}_processed.csv", index=False)


class TestCreateDataloadersNormalizerFitting:
    """Bug-1 regression: normalizers fitted on train only, then SHARED (not
    refit) into val/test."""

    def test_val_test_normalizers_are_the_same_fitted_objects_as_train(self, tmp_path):
        _write_synthetic_csvs(tmp_path)
        train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
            create_multi_stock_dataloaders(
                data_dir=str(tmp_path),
                seq_length=5,
                forecast_horizon=1,
                graph_method="correlation",
                batch_size=8,
                normalize=True,
                remove_outliers=True,
                n_std=3.0,
                data_augmentation=False,
            )

        assert train_ds.stock_names == val_ds.stock_names == test_ds.stock_names
        assert len(train_ds.target_normalizers) > 0
        for stock in train_ds.stock_names:
            # Identity: the exact same fitted object, not an independent refit.
            assert val_ds.target_normalizers[stock] is train_ds.target_normalizers[stock]
            assert test_ds.target_normalizers[stock] is train_ds.target_normalizers[stock]
            assert val_ds.feature_normalizers[stock] is train_ds.feature_normalizers[stock]
            assert test_ds.feature_normalizers[stock] is train_ds.feature_normalizers[stock]
            # And they are actually fitted.
            assert train_ds.target_normalizers[stock].mean is not None

    def test_splits_have_no_overlapping_dates(self, tmp_path):
        _write_synthetic_csvs(tmp_path)
        _tl, _vl, _te, (train_ds, val_ds, test_ds) = create_multi_stock_dataloaders(
            data_dir=str(tmp_path),
            seq_length=5,
            forecast_horizon=1,
            graph_method="correlation",
            batch_size=8,
            normalize=True,
            data_augmentation=False,
        )
        stock = train_ds.stock_names[0]
        train_dates = set(train_ds.stock_data[stock]["date"])
        val_dates = set(val_ds.stock_data[stock]["date"])
        test_dates = set(test_ds.stock_data[stock]["date"])
        assert train_dates.isdisjoint(val_dates)
        assert val_dates.isdisjoint(test_dates)
        assert train_dates.isdisjoint(test_dates)

    def test_batch_targets_are_normalized_scale(self, tmp_path):
        _write_synthetic_csvs(tmp_path)
        train_loader, *_ = create_multi_stock_dataloaders(
            data_dir=str(tmp_path),
            seq_length=5,
            forecast_horizon=1,
            graph_method="correlation",
            batch_size=8,
            normalize=True,
            data_augmentation=False,
        )
        x, _adj, y, _g = next(iter(train_loader))
        # Raw volatility is ~1e-2; normalized targets are O(1) with std well above
        # that. A std > 0.1 rules out the "still raw ~1e-3" failure mode.
        assert float(y.std()) > 0.1


class TestRealDataSmoke:
    """Real-data-sample smoke test per CLAUDE.md testing rules."""

    def test_real_data_dataloaders_normalized_and_split(self):
        data_dir = _ROOT / "data" / "processed"
        if not data_dir.exists():
            pytest.skip("data/processed/ not present in this environment")

        train_loader, val_loader, test_loader, (train_ds, val_ds, test_ds) = \
            create_multi_stock_dataloaders(
                data_dir=str(data_dir),
                seq_length=5,
                forecast_horizon=5,
                graph_method="correlation",
                batch_size=16,
                normalize=True,
                remove_outliers=True,
                n_std=3.0,
                data_augmentation=False,
            )

        assert len(train_loader) > 0 and len(val_loader) > 0 and len(test_loader) > 0

        # No overlapping dates across splits (reuse date-alignment assertion style).
        stock = train_ds.stock_names[0]
        train_dates = set(train_ds.stock_data[stock]["date"])
        val_dates = set(val_ds.stock_data[stock]["date"])
        test_dates = set(test_ds.stock_data[stock]["date"])
        assert train_dates.isdisjoint(val_dates)
        assert val_dates.isdisjoint(test_dates)

        # One batch through each split yields normalized-scale (not raw ~1e-3) targets.
        for loader in (train_loader, val_loader, test_loader):
            _x, _adj, y, _g = next(iter(loader))
            assert float(y.std()) > 0.1, "targets look un-normalized (raw ~1e-3 scale)"
