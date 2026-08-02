"""Smoke/shape test: S&P 500 processed data loads correctly into the
UNCHANGED VN30 MultiStockDataset (src/lstm_gat_hybrid/dataset.py).

Verifies the "thin wrapper, reuse unchanged" design decision (design.md §1):
- exactly the expected tickers are picked up (processing_summary.csv, which
  lacks date/parkinson_volatility columns, must be silently skipped, not
  mistaken for a ticker).
- at least 1 sequence is produced with the expected shapes.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
# baseline folder has a dash in its name -> not importable as a package;
# bootstrap sys.path to project root (per CLAUDE.md §3.F.4).
project_root = current_dir
for _ in range(3):
    project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)


def _write_processed_csv(path, n_rows=80, seed=0):
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    rng = np.random.RandomState(seed)
    vol = rng.uniform(0.0005, 0.02, n_rows)
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "parkinson_volatility": vol}).to_csv(
        path, index=False
    )


@pytest.fixture
def synthetic_sp500_dir(tmp_path):
    d = tmp_path / "processed_sp500_mini"
    d.mkdir()
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        _write_processed_csv(d / f"{ticker}_processed.csv", seed=i)
    # Non-ticker file that MUST be safely skipped (matches real
    # data/processed_sp500/processing_summary.csv).
    pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "num_records": [80, 80, 80]}).to_csv(
        d / "processing_summary.csv", index=False
    )
    return str(d)


class TestSp500DataLoadsIntoMultiStockDataset:
    def test_loads_exactly_expected_tickers_skips_summary_file(self, synthetic_sp500_dir):
        from src.lstm_gat_hybrid.dataset import MultiStockDataset

        ds = MultiStockDataset(
            data_dir=synthetic_sp500_dir,
            seq_length=10,
            forecast_horizon=1,
            graph_method="correlation",
            normalize=True,
            remove_outliers=False,
            data_augmentation=False,
        )

        assert set(ds.stock_names) == {"AAA", "BBB", "CCC"}
        assert "processing_summary" not in ds.stock_names

    def test_produces_sequences_with_expected_shapes(self, synthetic_sp500_dir):
        from src.lstm_gat_hybrid.dataset import MultiStockDataset

        seq_length = 10
        ds = MultiStockDataset(
            data_dir=synthetic_sp500_dir,
            seq_length=seq_length,
            forecast_horizon=1,
            graph_method="correlation",
            normalize=True,
            remove_outliers=False,
            data_augmentation=False,
        )

        assert len(ds) > 0
        x, adj_matrix, y, _ = ds[0]

        num_stocks = len(ds.stock_names)
        assert x.shape == (seq_length, num_stocks, 3)  # 3 HAR features
        assert adj_matrix.shape == (num_stocks, num_stocks)
        assert y.shape == (num_stocks,)

    def test_real_30_ticker_sp500_directory_loads(self):
        """Integration check against the REAL data/processed_sp500/ (30 tickers,
        this session's process_parkinson_pipeline run) -- not just synthetic
        data. Skips if the directory isn't present (e.g. fresh checkout before
        the pipeline has been run)."""
        real_dir = os.path.join(project_root, "data", "processed_sp500")
        if not os.path.isdir(real_dir):
            pytest.skip("data/processed_sp500 not present in this checkout")

        from src.lstm_gat_hybrid.dataset import MultiStockDataset

        ds = MultiStockDataset(
            data_dir=real_dir,
            seq_length=22,
            forecast_horizon=5,
            graph_method="correlation",
            normalize=True,
            remove_outliers=True,
            data_augmentation=False,
        )

        assert "processing_summary" not in ds.stock_names
        assert len(ds.stock_names) >= 25  # allow a few drops from outlier removal
        assert len(ds) > 0
