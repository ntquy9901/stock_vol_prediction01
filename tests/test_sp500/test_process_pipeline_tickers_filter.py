"""Tests for --tickers filter added to process_parkinson_pipeline.process_all_stocks.

Written to support restricting HAR processing to a specific ~30-ticker S&P 500
subset (for the upcoming LSTM-GNN baseline's k-NN graph) instead of all raw
files in data/raw/prices_sp500/.
"""
import os
import sys
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


def _write_raw_ohlcv(path, n_rows=40):
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    rng = np.random.RandomState(0)
    close = 100 + np.cumsum(rng.normal(0, 1, n_rows))
    high = close + rng.uniform(0.5, 2, n_rows)
    low = close - rng.uniform(0.5, 2, n_rows)
    open_ = close + rng.uniform(-1, 1, n_rows)
    volume = rng.randint(1000, 10000, n_rows)
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })
    df.to_csv(path, index=False)


class TestTickersFilter:
    def test_no_filter_processes_all_files(self, tmp_path):
        from src.common.process_parkinson_pipeline import process_all_stocks

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        raw_dir.mkdir()
        for t in ["AAA", "BBB", "CCC"]:
            _write_raw_ohlcv(raw_dir / f"{t}.csv")

        process_all_stocks(str(raw_dir), str(out_dir), tickers=None)

        produced = {os.path.splitext(f)[0].replace("_processed", "")
                    for f in os.listdir(out_dir) if f.endswith("_processed.csv")}
        assert produced == {"AAA", "BBB", "CCC"}

    def test_tickers_filter_restricts_to_requested_subset(self, tmp_path):
        from src.common.process_parkinson_pipeline import process_all_stocks

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        raw_dir.mkdir()
        for t in ["AAA", "BBB", "CCC", "DDD"]:
            _write_raw_ohlcv(raw_dir / f"{t}.csv")

        process_all_stocks(str(raw_dir), str(out_dir), tickers=["AAA", "CCC"])

        produced = {os.path.splitext(f)[0].replace("_processed", "")
                    for f in os.listdir(out_dir) if f.endswith("_processed.csv")}
        assert produced == {"AAA", "CCC"}

    def test_tickers_filter_missing_ticker_is_silently_skipped(self, tmp_path):
        from src.common.process_parkinson_pipeline import process_all_stocks

        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        raw_dir.mkdir()
        _write_raw_ohlcv(raw_dir / "AAA.csv")

        # "ZZZ" has no raw file -- must not raise, just produce fewer outputs.
        process_all_stocks(str(raw_dir), str(out_dir), tickers=["AAA", "ZZZ"])

        produced = {os.path.splitext(f)[0].replace("_processed", "")
                    for f in os.listdir(out_dir) if f.endswith("_processed.csv")}
        assert produced == {"AAA"}
