"""Validated raw price records and chronological per-ticker splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
import pandas as pd

from scaling import PreprocessorStore


_SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SampleKey:
    """Stable identity for one pooled forecast sample."""

    ticker_id: int
    ticker: str
    target_date: str


@dataclass(frozen=True)
class PooledSample:
    """Raw values retained before Task 2 preprocessing and window creation."""

    key: SampleKey
    x_price_raw: np.ndarray
    x_news: np.ndarray
    news_mask: np.ndarray
    y_raw: float
    y_model_raw: float | None = None
    y_eval_raw: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_price_raw", _readonly_array(self.x_price_raw, dtype=float))
        object.__setattr__(self, "x_news", _readonly_array(self.x_news, dtype=float))
        object.__setattr__(self, "news_mask", _readonly_array(self.news_mask, dtype=np.int8))


@dataclass(frozen=True)
class PooledManifest:
    """One shared P0-P3 sample set with stable eligibility decisions."""

    samples: Mapping[str, tuple[PooledSample, ...]]
    exclusions: Mapping[str, str]
    ticker_to_id: Mapping[str, int]
    preprocessing_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples",
            MappingProxyType({split: tuple(self.samples[split]) for split in _SPLIT_NAMES}),
        )
        object.__setattr__(self, "exclusions", MappingProxyType(dict(self.exclusions)))
        object.__setattr__(self, "ticker_to_id", MappingProxyType(dict(self.ticker_to_id)))

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": 1,
            "ticker_to_id": dict(self.ticker_to_id),
            "exclusions": dict(self.exclusions),
            "preprocessing_hash": self.preprocessing_hash,
            "samples": {
                split: [_sample_to_dict(sample) for sample in self.samples[split]]
                for split in _SPLIT_NAMES
            },
        }
        payload["hashes"] = _manifest_hashes(payload)
        return payload

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str, preprocessors: PreprocessorStore) -> "PooledManifest":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        hashes = payload.pop("hashes", None)
        if hashes != _manifest_hashes(payload):
            raise ValueError("manifest hash validation failed")
        preprocessing_hash = _stable_hash(preprocessors.to_dict())
        if payload["preprocessing_hash"] != preprocessing_hash:
            raise ValueError("manifest preprocessing hash validation failed")
        return cls(
            samples={
                split: tuple(_sample_from_dict(sample) for sample in payload["samples"][split])
                for split in _SPLIT_NAMES
            },
            exclusions=dict(payload["exclusions"]),
            ticker_to_id={str(ticker): int(ticker_id) for ticker, ticker_id in payload["ticker_to_id"].items()},
            preprocessing_hash=preprocessing_hash,
        )


@dataclass(frozen=True)
class SplitFrames:
    """Validated raw split frames and the deterministic ticker vocabulary."""

    frames: dict[str, dict[str, pd.DataFrame]]
    ticker_to_id: dict[str, int]


def _validate_ratios(ratios: tuple[float, float, float]) -> None:
    if len(ratios) != len(_SPLIT_NAMES) or any(ratio <= 0 for ratio in ratios):
        raise ValueError("ratios must contain three positive values")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must sum to 1.0")


def _validated_dates(df: pd.DataFrame) -> pd.Series:
    if "date" not in df.columns:
        raise ValueError("price data must contain a date column")
    if df.empty:
        raise ValueError("price data must not be empty")

    dates = pd.to_datetime(df["date"], errors="raise")
    if dates.duplicated().any():
        raise ValueError("duplicate dates are not allowed")
    if not dates.is_monotonic_increasing:
        raise ValueError("dates must be monotonic increasing")
    return dates


def chronological_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, pd.DataFrame]:
    """Validate and split one ticker's rows in their supplied chronological order."""

    _validate_ratios(ratios)
    dates = _validated_dates(df)
    train_end = int(len(df) * ratios[0])
    val_end = train_end + int(len(df) * ratios[1])
    boundaries = (0, train_end, val_end, len(df))
    parts = {
        name: df.iloc[boundaries[index] : boundaries[index + 1]].copy()
        for index, name in enumerate(_SPLIT_NAMES)
    }
    if any(part.empty for part in parts.values()):
        raise ValueError("each chronological split must contain at least one row")

    for part in parts.values():
        part["date"] = dates.loc[part.index].to_numpy()
    return parts


def load_and_split_price_data(
    data_dir: Path | str,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SplitFrames:
    """Load all processed ticker files and split each ticker without reordering rows."""

    _validate_ratios(ratios)
    directory = Path(data_dir)
    paths = sorted(directory.glob("*_processed.csv"))
    if not paths:
        raise ValueError(f"no *_processed.csv price files found under {directory}")

    tickers = [path.stem.removesuffix("_processed") for path in paths]
    if len(set(tickers)) != len(tickers):
        raise ValueError("ticker file stems must be unique")

    frames = {
        ticker: chronological_split(pd.read_csv(path), ratios)
        for ticker, path in zip(tickers, paths, strict=True)
    }
    return SplitFrames(
        frames=frames,
        ticker_to_id={ticker: ticker_id for ticker_id, ticker in enumerate(tickers)},
    )


def build_ticker_samples(
    frame: pd.DataFrame,
    ticker: str,
    ticker_id: int,
    feature_order: tuple[str, ...] | None = None,
    seq_length: int = 22,
    horizon: int = 5,
) -> list[PooledSample]:
    """Create windows whose target is exactly ``horizon`` observations after the origin."""

    if seq_length <= 0 or horizon <= 0:
        raise ValueError("seq_length and horizon must be positive")
    if "date" not in frame or "parkinson_volatility" not in frame:
        raise ValueError("frame must contain date and parkinson_volatility")
    if feature_order is None:
        feature_columns = ["parkinson_volatility"]
    else:
        feature_columns = [f"feature_{name}" for name in feature_order]
        stale_columns = [
            column
            for column in frame.columns
            if column.startswith("feature_") and column not in feature_columns
        ]
        if stale_columns:
            raise ValueError(f"unexpected transformed feature columns: {stale_columns}")
        if any(column not in frame for column in feature_columns):
            raise ValueError("missing expected transformed feature columns")
    feature_values = frame[feature_columns].to_numpy(dtype=float)
    if feature_values.ndim != 2 or feature_values.shape[1] != len(feature_columns):
        raise ValueError("invalid transformed feature dimension")
    if not np.isfinite(feature_values).all():
        raise ValueError("transformed feature values must be finite")
    model_targets = frame.get("y_model_raw", frame["parkinson_volatility"])
    eval_targets = frame.get("y_eval_raw", frame["parkinson_volatility"])
    valid_count = len(frame) - seq_length - horizon + 1
    samples: list[PooledSample] = []
    for start in range(max(0, valid_count)):
        target_index = start + seq_length + horizon - 1
        target_date = pd.Timestamp(frame.iloc[target_index]["date"]).strftime("%Y-%m-%d")
        y_eval_raw = float(eval_targets.iloc[target_index])
        samples.append(
            PooledSample(
                key=SampleKey(ticker_id=ticker_id, ticker=ticker, target_date=target_date),
                x_price_raw=feature_values[start : start + seq_length],
                x_news=np.empty((seq_length, 0), dtype=float),
                news_mask=np.zeros(seq_length, dtype=np.int8),
                y_raw=y_eval_raw,
                y_model_raw=float(model_targets.iloc[target_index]),
                y_eval_raw=y_eval_raw,
            )
        )
    return samples


def build_pooled_manifest(
    split_frames: SplitFrames,
    preprocessors: PreprocessorStore,
    seq_length: int = 22,
    horizon: int = 5,
) -> PooledManifest:
    """Transform each raw split using train-fitted state and pool eligible ticker samples."""

    samples: dict[str, list[PooledSample]] = {name: [] for name in _SPLIT_NAMES}
    exclusions: dict[str, str] = {}
    for ticker, ticker_id in sorted(split_frames.ticker_to_id.items(), key=lambda item: item[1]):
        ticker_samples: dict[str, list[PooledSample]] = {}
        for split_name in _SPLIT_NAMES:
            preprocessor = preprocessors.get(ticker_id)
            transformed = preprocessor.transform_frame(split_frames.frames[ticker][split_name])
            ticker_samples[split_name] = build_ticker_samples(
                transformed,
                ticker,
                ticker_id,
                feature_order=preprocessor.feature_order,
                seq_length=seq_length,
                horizon=horizon,
            )
        if any(not ticker_samples[name] for name in _SPLIT_NAMES):
            exclusions[ticker] = "insufficient windows in every split"
            continue
        for split_name in _SPLIT_NAMES:
            samples[split_name].extend(ticker_samples[split_name])
    for split_name in _SPLIT_NAMES:
        samples[split_name].sort(key=lambda sample: (sample.key.target_date, sample.key.ticker_id))
    return PooledManifest(
        samples={split: tuple(split_samples) for split, split_samples in samples.items()},
        exclusions=exclusions,
        ticker_to_id=dict(split_frames.ticker_to_id),
        preprocessing_hash=_stable_hash(preprocessors.to_dict()),
    )


def _sample_to_dict(sample: PooledSample) -> dict[str, object]:
    return {
        "ticker_id": sample.key.ticker_id,
        "ticker": sample.key.ticker,
        "target_date": sample.key.target_date,
        "x_price_raw": sample.x_price_raw.tolist(),
        "x_news": sample.x_news.tolist(),
        "news_mask": sample.news_mask.tolist(),
        "y_raw": sample.y_raw,
        "y_model_raw": sample.y_model_raw,
        "y_eval_raw": sample.y_eval_raw,
    }


def _sample_from_dict(value: dict[str, object]) -> PooledSample:
    return PooledSample(
        key=SampleKey(int(value["ticker_id"]), str(value["ticker"]), str(value["target_date"])),
        x_price_raw=np.asarray(value["x_price_raw"], dtype=float),
        x_news=np.asarray(value["x_news"], dtype=float),
        news_mask=np.asarray(value["news_mask"], dtype=np.int8),
        y_raw=float(value["y_raw"]),
        y_model_raw=float(value["y_model_raw"]),
        y_eval_raw=float(value["y_eval_raw"]),
    )


def _manifest_hashes(payload: dict[str, object]) -> dict[str, str]:
    samples = payload["samples"]
    return {
        "sample_ids": _stable_hash(
            {
                split: [(sample["ticker_id"], sample["target_date"]) for sample in split_samples]
                for split, split_samples in samples.items()
            }
        ),
        "raw_price_inputs": _stable_hash(
            {split: [sample["x_price_raw"] for sample in split_samples] for split, split_samples in samples.items()}
        ),
        "news_inputs_masks": _stable_hash(
            {
                split: [(sample["x_news"], sample["news_mask"]) for sample in split_samples]
                for split, split_samples in samples.items()
            }
        ),
        "raw_targets": _stable_hash(
            {split: [sample["y_eval_raw"] for sample in split_samples] for split, split_samples in samples.items()}
        ),
        "preprocessing": str(payload["preprocessing_hash"]),
        "manifest": _stable_hash(payload),
    }


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readonly_array(values: np.ndarray, dtype: type[np.floating] | type[np.int8]) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array
