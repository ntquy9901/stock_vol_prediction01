"""Tests for --forecast_horizon target-column correctness (feature_merger + cross_market).

Test-first: written against the NEW `horizon` param before it exists in the source files.
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


@pytest.fixture
def sample_processed_data():
    dates = pd.date_range("2011-01-03", periods=100, freq="B")
    vol = np.arange(100, dtype=np.float64) / 1000.0  # known ramp: 0.000, 0.001, ..., 0.099
    return pd.DataFrame({"date": dates, "parkinson_volatility": vol})


class TestFeatureMergerHorizon:
    @pytest.mark.parametrize("horizon", [1, 10])
    def test_target_column_matches_shift(self, sample_processed_data, tmp_path, horizon):
        from src.common.feature_merger import merge_features

        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir)
        sample_processed_data.to_csv(os.path.join(processed_dir, "AAPL_processed.csv"), index=False)

        output_dir = str(tmp_path / "enhanced")
        result = merge_features(
            ticker="AAPL",
            processed_dir=processed_dir,
            market_data_dir=str(tmp_path / "market"),
            sentiment_dir=str(tmp_path / "sentiment"),
            output_dir=output_dir,
            feature_set="har",
            horizon=horizon,
        )

        df = pd.read_csv(result, parse_dates=["Date"], index_col="Date")
        target_col = f"target_{horizon}d"
        assert target_col in df.columns

        # Reload raw (unfiltered) to compare against known ramp values.
        raw = pd.read_csv(os.path.join(processed_dir, "AAPL_processed.csv"))
        expected = raw["parkinson_volatility"].shift(-horizon)
        # merge_features drops NaN target rows; compare on overlapping dates.
        raw["date"] = pd.to_datetime(raw["date"])
        expected_by_date = pd.Series(expected.values, index=raw["date"])
        for date, actual_val in df[target_col].items():
            assert abs(actual_val - expected_by_date.loc[date]) < 1e-9

    def test_default_horizon_still_5_backward_compatible(self, sample_processed_data, tmp_path):
        from src.common.feature_merger import merge_features

        processed_dir = str(tmp_path / "processed")
        os.makedirs(processed_dir)
        sample_processed_data.to_csv(os.path.join(processed_dir, "AAPL_processed.csv"), index=False)

        result = merge_features(
            ticker="AAPL",
            processed_dir=processed_dir,
            market_data_dir=str(tmp_path / "market"),
            sentiment_dir=str(tmp_path / "sentiment"),
            output_dir=str(tmp_path / "enhanced"),
            feature_set="har",
        )
        df = pd.read_csv(result)
        assert "target_5d" in df.columns


class TestCrossMarketHorizon:
    @pytest.mark.parametrize("horizon", [1, 10])
    def test_load_market_data_target_matches_shift(self, tmp_path, horizon):
        from src.experiments.sp500.cross_market_experiment import load_market_data

        processed_dir = str(tmp_path / "processed_sp500")
        os.makedirs(processed_dir)
        dates = pd.date_range("2011-01-03", periods=100, freq="B")
        vol = np.arange(100, dtype=np.float64) / 1000.0
        df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "parkinson_volatility": vol})
        df.to_csv(os.path.join(processed_dir, "AAPL_processed.csv"), index=False)

        import src.experiments.sp500.cross_market_experiment as cme
        original_dirs = cme.MARKET_PROCESSED_DIRS.copy()
        cme.MARKET_PROCESSED_DIRS["sp500"] = processed_dir
        try:
            ticker_dfs = load_market_data("sp500", tickers=["AAPL"], horizon=horizon)
        finally:
            cme.MARKET_PROCESSED_DIRS.clear()
            cme.MARKET_PROCESSED_DIRS.update(original_dirs)

        result_df = ticker_dfs["AAPL"] if isinstance(ticker_dfs, dict) else ticker_dfs
        target_col = f"target_{horizon}d"
        assert target_col in result_df.columns
        expected = vol[horizon:]  # shift(-horizon) drops last `horizon` rows after dropna
        actual = result_df[target_col].values
        assert len(actual) == len(expected)
        np.testing.assert_allclose(actual, expected)
