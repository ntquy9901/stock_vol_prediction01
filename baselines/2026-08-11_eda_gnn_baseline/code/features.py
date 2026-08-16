"""Leakage-safe extended node features for the EDA-recommended GNN.

Adds two features to the pilot's HAR(3) node features, in the nested order the ablation ladder
slices by:

    (parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20)

* ``market_pk`` -- cross-sectional MEDIAN of ``sqrt(parkinson_volatility)`` over all present tickers
  at date ``t`` (the market factor itself). Contemporaneous (uses only column ``t``) -> leakage-safe.
* ``volume_zscore_20`` -- trailing rolling z-score ``(log1p(volume) - roll20.mean) / roll20.std``.
  Trailing window only -> causal/leakage-safe. Tickers without an OHLCV volume series (e.g. LPB) get
  a neutral ``0.0`` so their windows stay eligible.

Both features add no leading NaN beyond the monthly-HAR ``rolling(22)`` warm-up (vol_z is
``rolling(20)``; market_pk has none), so the eligible-window set is byte-identical to the 3-feature
pilot and per-column standardisation leaves the first three feature columns bit-identical. This keeps
E0 == the pilot HAR P0 and the val/test observation set aligned with ``ladder_consistent``.

Imports the pilot pipeline read-only (hard isolation): nothing here mutates a pilot module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import SplitFrames, _SPLIT_NAMES
from scaling import ArrayStandardizer, PreprocessorStore, _har_features

# The two extra raw feature columns, in nested-ladder order (MarketPK first: the biggest lever).
MARKET_PK_COLUMN = "market_pk"
VOLUME_ZSCORE_COLUMN = "volume_zscore_20"
EXTRA_FEATURE_COLUMNS = (MARKET_PK_COLUMN, VOLUME_ZSCORE_COLUMN)
_BASE_HAR_ORDER = ("parkinson_volatility", "har_weekly", "har_monthly")
_VOLUME_WINDOW = 20


def _local_date_key(values) -> pd.Series:
    """Normalise dates to ``YYYY-MM-DD`` local wall-time strings (tz stripped, not shifted)."""

    parsed = pd.to_datetime(pd.Series(values).reset_index(drop=True))
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.strftime("%Y-%m-%d")


def _full_series(split_frames: SplitFrames, ticker: str) -> pd.DataFrame:
    """Concatenate a ticker's three chronological split frames back into one ordered frame."""

    parts = [split_frames.frames[ticker][name] for name in _SPLIT_NAMES]
    full = pd.concat(parts, axis=0)
    full = full.copy()
    full["date"] = pd.to_datetime(full["date"])
    return full


def volume_zscore_series(dates: pd.Series, price_dir: Path | str, ticker: str) -> pd.Series:
    """Trailing rolling-20 z-score of ``log1p(volume)`` aligned to ``dates`` (0.0 where no volume).

    The z-score is computed on the ticker's full chronological volume series (so validation/test
    rows use a trailing window that reaches into the prior split), then aligned to ``dates``. A
    ticker with no ``*_ohlcv.csv`` (e.g. LPB) yields all zeros -- a neutral shock that keeps its
    windows eligible without inventing volume data.
    """

    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    path = Path(price_dir) / f"{ticker}_ohlcv.csv"
    if not path.exists():
        return pd.Series(np.zeros(len(dates), dtype=float), index=dates.to_numpy())
    raw = pd.read_csv(path)
    # Align on the local calendar date. Some tickers (e.g. VPB/VRE) store tz-aware wall-midnight
    # timestamps; strip the tz WITHOUT shifting the wall time so the date is not moved a day.
    raw_key = _local_date_key(raw["date"])
    volume_by_key = raw.assign(_key=raw_key).groupby("_key")["volume"].last()
    volume = volume_by_key.reindex(_local_date_key(dates).to_numpy())
    log_volume = np.log1p(volume.to_numpy(dtype=float))
    series = pd.Series(log_volume)
    mean = series.rolling(_VOLUME_WINDOW).mean()
    std = series.rolling(_VOLUME_WINDOW).std()
    zscore = (series - mean) / std.replace(0.0, np.nan)
    # A flat/zero-volume window (std == 0, e.g. pre-listing backfilled zeros) or a missing volume
    # gives a non-finite z; map it to 0.0 (a neutral "no shock"). This is causal (trailing window
    # only) and keeps those rows eligible so the observation set stays byte-identical to the HAR
    # pipeline, whose monthly-HAR rolling(22) warm-up remains the binding constraint.
    zscore = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.Series(zscore.to_numpy(dtype=float), index=dates.to_numpy())


def market_pk_series(split_frames: SplitFrames) -> pd.Series:
    """Cross-sectional median of ``sqrt(parkinson_volatility)`` across tickers, indexed by date."""

    columns: dict[str, pd.Series] = {}
    for ticker in split_frames.frames:
        full = _full_series(split_frames, ticker)
        pk = np.sqrt(full["parkinson_volatility"].to_numpy(dtype=float))
        columns[ticker] = pd.Series(pk, index=pd.DatetimeIndex(full["date"]))
    wide = pd.DataFrame(columns)
    median = wide.median(axis=1, skipna=True)
    return median.sort_index()


def augment_split_frames(split_frames: SplitFrames, price_dir: Path | str) -> SplitFrames:
    """Return a copy of ``split_frames`` with market_pk + volume_zscore_20 columns added per split.

    Both columns are computed on each ticker's full chronological series and mapped back to every
    split by date, so a split boundary never truncates a trailing window or a contemporaneous
    market value.
    """

    market = market_pk_series(split_frames)
    market_by_date = {pd.Timestamp(date): float(value) for date, value in market.items()}
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for ticker in split_frames.frames:
        full = _full_series(split_frames, ticker)
        vol_z = volume_zscore_series(full["date"], price_dir, ticker)
        vol_z_by_date = {pd.Timestamp(date): float(value) for date, value in vol_z.items()}
        frames[ticker] = {}
        for split in _SPLIT_NAMES:
            part = split_frames.frames[ticker][split].copy()
            keys = pd.to_datetime(part["date"]).map(pd.Timestamp)
            part[MARKET_PK_COLUMN] = keys.map(market_by_date).to_numpy(dtype=float)
            part[VOLUME_ZSCORE_COLUMN] = keys.map(vol_z_by_date).to_numpy(dtype=float)
            frames[ticker][split] = part
    return SplitFrames(frames=frames, ticker_to_id=dict(split_frames.ticker_to_id))


@dataclass(frozen=True)
class ExtendedTickerPreprocessor:
    """HAR(3) + extra raw features, per-column standardised on train rows only.

    Mirrors ``scaling.TickerPreprocessor`` but appends pre-computed extra feature columns
    (``market_pk`` / ``volume_zscore_20``) after the three HAR features. The clip bounds, HAR
    construction, and target scaler are identical to the pilot, so the first three feature columns
    are bit-identical to the 3-feature pilot.
    """

    feature_order: tuple[str, ...]
    extra_columns: tuple[str, ...]
    target_column: str
    lower_bound: float
    upper_bound: float
    feature_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer

    @classmethod
    def fit(cls, train_frame: pd.DataFrame, extra_columns: tuple[str, ...]) -> "ExtendedTickerPreprocessor":
        target = "parkinson_volatility"
        if target not in train_frame:
            raise ValueError(f"missing target column: {target}")
        for column in extra_columns:
            if column not in train_frame:
                raise ValueError(f"missing extra feature column: {column}")
        raw_target = train_frame[target].to_numpy(dtype=float)
        if raw_target.size == 0 or not np.isfinite(raw_target).all():
            raise ValueError("train target values must be non-empty and finite")
        mean = float(raw_target.mean())
        std = float(raw_target.std())
        lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        clipped = np.clip(raw_target, lower_bound, upper_bound)
        features = cls._stack(clipped, train_frame, extra_columns)
        valid_rows = np.isfinite(features).all(axis=1)
        return cls(
            feature_order=(*_BASE_HAR_ORDER, *extra_columns),
            extra_columns=tuple(extra_columns),
            target_column=target,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            feature_scaler=ArrayStandardizer().fit(features[valid_rows]),
            target_scaler=ArrayStandardizer().fit(clipped[valid_rows]),
        )

    @staticmethod
    def _stack(clipped: np.ndarray, frame: pd.DataFrame, extra_columns: tuple[str, ...]) -> np.ndarray:
        har = _har_features(clipped)
        if not extra_columns:
            return har
        extras = frame[list(extra_columns)].to_numpy(dtype=float)
        return np.column_stack((har, extras))

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.target_column not in frame:
            raise ValueError(f"missing target column: {self.target_column}")
        result = frame.copy()
        y_eval = result[self.target_column].to_numpy(dtype=float)
        if not np.isfinite(y_eval).all():
            raise ValueError("split target values must be finite")
        y_model = np.clip(y_eval, self.lower_bound, self.upper_bound)
        features = self._stack(y_model, result, self.extra_columns)
        valid_rows = np.isfinite(features).all(axis=1)
        # Guard the observation-set invariant: the extra features must not drop (or keep) any row
        # the 3-feature HAR pipeline would not, so E0 stays comparable to the pilot HAR P0 and the
        # val/test set matches ladder_consistent (finding M2).
        har_valid = np.isfinite(features[:, :3]).all(axis=1)
        if not np.array_equal(valid_rows, har_valid):
            raise ValueError("extra features changed the HAR-eligible window set (observation drift)")
        result = result.loc[valid_rows].copy()
        normalized = self.feature_scaler.transform(features[valid_rows])
        result["y_model_raw"] = y_model[valid_rows]
        result["y_eval_raw"] = y_eval[valid_rows]
        for index, name in enumerate(self.feature_order):
            result[f"feature_{name}"] = normalized[:, index]
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_order": list(self.feature_order),
            "extra_columns": list(self.extra_columns),
            "target_column": self.target_column,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "feature_scaler": self.feature_scaler.to_dict(),
            "target_scaler": self.target_scaler.to_dict(),
        }


def build_extended_store(
    augmented_frames: SplitFrames, extra_columns: tuple[str, ...] = EXTRA_FEATURE_COLUMNS
) -> PreprocessorStore:
    """Fit one ExtendedTickerPreprocessor per ticker on its TRAIN split (extra columns included)."""

    preprocessors = {
        ticker_id: ExtendedTickerPreprocessor.fit(augmented_frames.frames[ticker]["train"], extra_columns)
        for ticker, ticker_id in augmented_frames.ticker_to_id.items()
    }
    return PreprocessorStore(preprocessors)
