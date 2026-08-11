"""Leakage + parity tests for the extended EDA node features."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import features
from data import SplitFrames, chronological_split
from scaling import TickerPreprocessor


def _split_frames_from(series: dict[str, pd.DataFrame]) -> SplitFrames:
    frames = {ticker: chronological_split(df) for ticker, df in series.items()}
    ticker_to_id = {ticker: index for index, ticker in enumerate(sorted(series))}
    return SplitFrames(frames=frames, ticker_to_id=ticker_to_id)


def _synthetic_pk_frame(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    pk = np.abs(rng.normal(1e-3, 3e-4, size=n)) + 1e-5
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "parkinson_volatility": pk})


def test_volume_zscore_is_causal(tmp_path):
    """Altering a FUTURE volume must not change the trailing z-score at earlier dates."""

    dates = pd.bdate_range("2015-01-01", periods=60)
    volume = np.linspace(1000, 2000, 60)
    frame = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": 1, "high": 1, "low": 1,
                          "close": 1, "volume": volume})
    frame.to_csv(tmp_path / "AAA_ohlcv.csv", index=False)
    base = features.volume_zscore_series(pd.Series(dates), tmp_path, "AAA")

    spiked = frame.copy()
    spiked.loc[50:, "volume"] = 9_999_999  # future spike from index 50 onward
    spiked.to_csv(tmp_path / "AAA_ohlcv.csv", index=False)
    after = features.volume_zscore_series(pd.Series(dates), tmp_path, "AAA")

    early = slice(0, 50)
    assert np.allclose(base.to_numpy()[early], after.to_numpy()[early], equal_nan=True)
    # And the spike DID change a later value (guards against a no-op test).
    assert not np.allclose(base.to_numpy()[55], after.to_numpy()[55], equal_nan=True)


def test_volume_zscore_missing_ticker_is_zero(tmp_path):
    dates = pd.bdate_range("2015-01-01", periods=30)
    series = features.volume_zscore_series(pd.Series(dates), tmp_path, "NOFILE")
    assert np.array_equal(series.to_numpy(), np.zeros(30))


def test_market_pk_is_contemporaneous():
    """MarketPK at t uses only column t: altering a future PK leaves earlier MarketPK unchanged."""

    split_frames = _split_frames_from({t: _synthetic_pk_frame(i) for i, t in enumerate(("AAA", "BBB", "CCC"))})
    base = features.market_pk_series(split_frames)

    bumped = {t: split_frames.frames[t] for t in split_frames.frames}
    # Bump AAA's very last PK (a future date) massively.
    last_split = bumped["AAA"]["test"].copy()
    last_split.iloc[-1, last_split.columns.get_loc("parkinson_volatility")] *= 100
    bumped["AAA"] = {**bumped["AAA"], "test": last_split}
    bumped_frames = SplitFrames(frames=bumped, ticker_to_id=dict(split_frames.ticker_to_id))
    after = features.market_pk_series(bumped_frames)

    common = base.index.intersection(after.index)[:-1]  # every date except the bumped last one
    assert np.allclose(base.loc[common].to_numpy(), after.loc[common].to_numpy())


def test_market_pk_equals_cross_sectional_median():
    split_frames = _split_frames_from({t: _synthetic_pk_frame(i) for i, t in enumerate(("AAA", "BBB", "CCC"))})
    market = features.market_pk_series(split_frames)
    # Recompute the median of sqrt(PK) at one date directly.
    date = market.index[100]
    values = []
    for ticker in split_frames.frames:
        full = features._full_series(split_frames, ticker).set_index("date")
        values.append(np.sqrt(full.loc[date, "parkinson_volatility"]))
    assert market.loc[date] == pytest.approx(float(np.median(values)))


def _real_train_frame() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[3]
    frame = pd.read_csv(root / "data" / "processed" / "ACB_processed.csv")
    return frame.iloc[:1500].copy()


def test_extended_first_three_features_match_pilot():
    """With finite extras, the first three feature columns are bit-identical to the 3-feature pilot."""

    frame = _real_train_frame()
    frame["market_pk"] = 0.5
    frame["volume_zscore_20"] = 0.1
    pilot = TickerPreprocessor.fit(frame, ["parkinson_volatility"], "parkinson_volatility")
    extended = features.ExtendedTickerPreprocessor.fit(frame, features.EXTRA_FEATURE_COLUMNS)
    assert np.allclose(extended.feature_scaler.mean[:3], pilot.feature_scaler.mean)
    assert np.allclose(extended.feature_scaler.std[:3], pilot.feature_scaler.std)
    assert extended.feature_order[:3] == pilot.feature_order


def test_extended_transform_preserves_observation_set():
    """Adding finite extras does not drop or add rows vs the pilot HAR transform (obs set stable)."""

    frame = _real_train_frame()
    frame["market_pk"] = np.linspace(0.4, 0.6, len(frame))
    frame["volume_zscore_20"] = np.linspace(-1.0, 1.0, len(frame))
    pilot = TickerPreprocessor.fit(frame, ["parkinson_volatility"], "parkinson_volatility")
    extended = features.ExtendedTickerPreprocessor.fit(frame, features.EXTRA_FEATURE_COLUMNS)
    pilot_rows = pilot.transform_frame(frame)
    extended_rows = extended.transform_frame(frame)
    assert len(pilot_rows) == len(extended_rows)
    assert np.array_equal(pilot_rows["y_eval_raw"].to_numpy(), extended_rows["y_eval_raw"].to_numpy())
    assert np.allclose(pilot_rows["feature_parkinson_volatility"].to_numpy(),
                       extended_rows["feature_parkinson_volatility"].to_numpy())
