"""Regression tests for 2 bugs confirmed present in src/lstm_gat_hybrid/dataset.py
(UNCHANGED, read-only import per CLAUDE.md §3.F.3) via the VN30 project's
2026-08-02 audit (AUD-001, AUD-002), verified to also apply to this baseline
since it imports the exact same dataset.py.

These tests are written RED-first (TDD): they encode the CORRECT behavior and
currently FAIL against dataset.py's actual behavior. They are marked
xfail(strict=True) rather than left red, per explicit 2026-08-02 decision:
fix dataset.py on the VN30 master project first (it is shared, unmodified,
read-only infra -- CLAUDE.md §3.F.3), then merge/cherry-pick that fix into
this branch, instead of fixing it independently here. strict=True means
these must flip to XPASS (and have their marker removed) once that fix
lands -- an unexpected pass is reported as a failure, so the flip can't be
missed. See docs/reports/2026-08-02_*_sp500_audit_applicability_and_plan.md.

AUD-001 (leakage): MultiStockDataset._initialize_normalizers() fits
VolatilityNormalizer on the ENTIRE per-stock DataFrame in __init__, before
create_multi_stock_dataloaders() ever computes a train/val/test split (the
split is applied afterward via torch.utils.data.Subset, which only restricts
__getitem__ -- it does not affect what the normalizer was fit on, because
train/val/test are 3 SEPARATE full MultiStockDataset instances that each
independently fit on 100% of the data).

AUD-002 (misalignment): _load_multi_stock_data() calls remove_outliers() per
ticker independently, so two tickers can end up with a different number of
(and different) rows remaining. _create_sequences() then indexes every
ticker's remaining frame by the same positional `.iloc[i:...]`, with no date
join -- position i is not guaranteed to be the same calendar date across
tickers once outlier removal has desynced them.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
for _ in range(3):
    project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)

from src.lstm_gat_hybrid.dataset import MultiStockDataset, create_multi_stock_dataloaders  # noqa: E402


def _write_csv(path, dates, volatility):
    pd.DataFrame({"date": dates, "parkinson_volatility": volatility}).to_csv(path, index=False)


class TestNormalizerLeakage:
    """AUD-001: target normalizer must be fit on the train split only."""

    @pytest.mark.xfail(
        reason="AUD-001: dataset.py fits normalizers on 100% of data before "
               "split; fix pending on VN30 master, then merge here (2026-08-02).",
        strict=True,
    )
    def test_train_normalizer_is_not_pulled_by_val_test_spike(self, tmp_path):
        n_rows = 60
        dates = pd.bdate_range("2020-01-01", periods=n_rows).strftime("%Y-%m-%d")

        # First 40 rows (train-ish range): tiny, near-constant volatility.
        # Last 20 rows (val+test range): a 5.0 spike, ~50,000x the train scale.
        volatility = np.concatenate([
            0.0001 + 0.0000001 * np.arange(40),
            np.full(20, 5.0),
        ])
        _write_csv(tmp_path / "AAA_processed.csv", dates, volatility)

        _, _, _, datasets = create_multi_stock_dataloaders(
            data_dir=str(tmp_path),
            seq_length=5,
            forecast_horizon=1,
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            remove_outliers=False,  # isolate leakage from the alignment bug (AUD-002)
            data_augmentation=False,
        )
        train_dataset = datasets[0]
        fitted_mean = train_dataset.dataset.target_normalizers["AAA"].mean

        assert fitted_mean < 0.1, (
            f"target_normalizer mean={fitted_mean!r} was pulled toward the val/test "
            "spike (~5.0) instead of reflecting only the train-range values "
            "(~0.0001) -- the normalizer was fit on the full dataset, not the "
            "train split (AUD-001)."
        )


class TestCrossStockDateAlignment:
    """AUD-002: sequences must align stocks by calendar date, not row position."""

    @pytest.mark.xfail(
        reason="AUD-002: dataset.py removes outliers per-ticker then stacks by "
               "row position, no date join; fix pending on VN30 master, then "
               "merge here (2026-08-02).",
        strict=True,
    )
    def test_same_sequence_position_is_same_date_across_stocks(self, tmp_path):
        n_rows = 40
        dates = pd.bdate_range("2020-01-01", periods=n_rows).strftime("%Y-%m-%d")

        # AAA: one deliberate outlier at row 5 (huge spike) -> remove_outliers()
        # drops exactly that row, desyncing AAA's remaining rows from BBB's.
        vol_aaa = 0.0001 + 0.0000001 * np.arange(n_rows)
        vol_aaa[5] = 1000.0
        _write_csv(tmp_path / "AAA_processed.csv", dates, vol_aaa)

        # BBB: no outliers, all rows kept.
        vol_bbb = 0.0002 + 0.0000001 * np.arange(n_rows)
        _write_csv(tmp_path / "BBB_processed.csv", dates, vol_bbb)

        dataset = MultiStockDataset(
            data_dir=str(tmp_path),
            seq_length=5,
            forecast_horizon=1,
            normalize=False,
            remove_outliers=True,
            n_std=3.0,
            data_augmentation=False,
        )

        aaa_dates = dataset.stock_data_with_har["AAA"]["date"].reset_index(drop=True)
        bbb_dates = dataset.stock_data_with_har["BBB"]["date"].reset_index(drop=True)

        assert len(aaa_dates) == n_rows - 1, "expected exactly 1 row removed from AAA"
        assert len(bbb_dates) == n_rows, "expected 0 rows removed from BBB"

        # _create_sequences() reads both frames with the same positional
        # `.iloc[i:...]` -- so position i must be the same date in both, or
        # the panel is temporally incoherent (AUD-002).
        position_after_removal = 10
        assert aaa_dates.iloc[position_after_removal] == bbb_dates.iloc[position_after_removal], (
            f"position {position_after_removal}: AAA date="
            f"{aaa_dates.iloc[position_after_removal]!r} vs BBB date="
            f"{bbb_dates.iloc[position_after_removal]!r} -- per-ticker outlier "
            "removal desynced the two frames, but _create_sequences() stacks "
            "them by row position with no date join (AUD-002)."
        )
