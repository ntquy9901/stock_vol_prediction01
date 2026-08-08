"""Validated raw price records and chronological per-ticker splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

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
    input_dates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_price_raw", _readonly_array(self.x_price_raw, dtype=np.float32))
        object.__setattr__(self, "x_news", _readonly_array(self.x_news, dtype=np.float32))
        object.__setattr__(self, "news_mask", _readonly_array(self.news_mask, dtype=np.int8))
        normalized_dates = tuple(pd.Timestamp(date).strftime("%Y-%m-%d") for date in self.input_dates)
        if normalized_dates and len(normalized_dates) != len(self.x_price_raw):
            raise ValueError("input_dates must match the price sequence length")
        object.__setattr__(self, "input_dates", normalized_dates)


@dataclass(frozen=True)
class NewsPanel:
    """Validated effective-trading-date news vectors keyed by ticker and date."""

    values: Mapping[tuple[str, str], np.ndarray]
    feature_cols: tuple[str, ...]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        values = {
            (str(ticker), str(date)): _readonly_array(vector, dtype=np.float32)
            for (ticker, date), vector in self.values.items()
        }
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "feature_cols", tuple(self.feature_cols))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def keys(self):
        return self.values.keys()


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

    def content_hash(self, split: str) -> str:
        """Return a deterministic digest of one split's data and preprocessing state."""

        if split not in _SPLIT_NAMES:
            raise ValueError(f"unknown split: {split}")
        return _stable_hash(
            {
                "eligibility_raw_targets": _eligibility_target_hash(self.samples[split]),
                "price_tensors": _tensor_hash(self.samples[split], "x_price_raw"),
                "news_tensors_masks": _news_tensor_hash(self.samples[split]),
                "preprocessing": self.preprocessing_hash,
            }
        )

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
                input_dates=tuple(
                    pd.Timestamp(date).strftime("%Y-%m-%d")
                    for date in frame.iloc[start : start + seq_length]["date"]
                ),
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
        "input_dates": list(sample.input_dates),
    }


def _sample_from_dict(value: dict[str, object]) -> PooledSample:
    return PooledSample(
        key=SampleKey(int(value["ticker_id"]), str(value["ticker"]), str(value["target_date"])),
        x_price_raw=np.asarray(value["x_price_raw"], dtype=float),
        x_news=np.asarray(value["x_news"], dtype=float),
        news_mask=np.asarray(value["news_mask"], dtype=np.int8),
        y_raw=float(value["y_raw"]),
        y_model_raw=None if value["y_model_raw"] is None else float(value["y_model_raw"]),
        y_eval_raw=None if value["y_eval_raw"] is None else float(value["y_eval_raw"]),
        input_dates=tuple(str(date) for date in value.get("input_dates", [])),
    )


def _manifest_hashes(payload: dict[str, object]) -> dict[str, str]:
    samples = payload["samples"]
    return {
        "eligibility_raw_targets": _stable_hash({
            split: [
                (sample["ticker_id"], sample["ticker"], sample["target_date"],
                 sample["input_dates"],
                 sample["y_raw"] if sample["y_eval_raw"] is None else sample["y_eval_raw"])
                for sample in split_samples
            ]
            for split, split_samples in samples.items()
        }),
        "price_tensors": _stable_hash({
            split: [
                _canonical_array_hash(np.asarray(sample["x_price_raw"], dtype=np.float32))
                for sample in split_samples
            ]
            for split, split_samples in samples.items()
        }),
        "news_tensors_masks": _stable_hash({
            split: [
                (_canonical_array_hash(np.asarray(sample["x_news"], dtype=np.float32)),
                 _canonical_array_hash(np.asarray(sample["news_mask"], dtype=np.int8)))
                for sample in split_samples
            ]
            for split, split_samples in samples.items()
        }),
        "preprocessing": str(payload["preprocessing_hash"]),
        "manifest": _stable_hash(payload),
    }


def _stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _readonly_array(values: np.ndarray, dtype: type[np.floating] | type[np.int8]) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("sample arrays must be finite")
    array.setflags(write=False)
    return array


def load_effective_news_panel(
    path: Path | str,
    eligible_train_cutoff: str | Mapping[str, str] | None = None,
    tickers: Iterable[str] | None = None,
) -> NewsPanel:
    """Load a precomputed, causally aligned news panel without refitting it."""

    panel_path = Path(path)
    selected_tickers = None if tickers is None else sorted({str(ticker) for ticker in tickers})
    frame = pd.read_parquet(
        panel_path,
        filters=None if selected_tickers is None else [("ticker", "in", selected_tickers)],
    )
    required = {"ticker", "date"}
    if not required.issubset(frame.columns):
        raise ValueError("news panel must contain ticker and date columns")
    feature_cols = tuple(sorted(column for column in frame.columns if column not in required))
    if not feature_cols:
        raise ValueError("news panel must contain feature columns")
    normalized = frame.copy()
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip()
    if (normalized["ticker"] == "").any():
        raise ValueError("news panel ticker must not be empty")
    try:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime("%Y-%m-%d")
    except (TypeError, ValueError) as error:
        raise ValueError("news panel dates must be parseable") from error
    if normalized.duplicated(["ticker", "date"]).any():
        raise ValueError("news panel (ticker, date) keys must be unique")
    values = normalized.loc[:, feature_cols].to_numpy(dtype=np.float32)
    missing_rows = np.isnan(values).all(axis=1)
    if np.isinf(values).any():
        raise ValueError("news panel features must be finite")
    values = np.nan_to_num(values, nan=0.0)
    provenance_path = panel_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    _validate_news_provenance(provenance, eligible_train_cutoff)
    return NewsPanel(
        values={
            (str(row.ticker), str(row.date)): values[index]
            for index, row in enumerate(normalized.loc[:, ["ticker", "date"]].itertuples(index=False))
            if not missing_rows[index]
        },
        feature_cols=feature_cols,
        provenance=provenance,
    )


def attach_news(
    samples: list[PooledSample], panel: NewsPanel, feature_cols: list[str] | tuple[str, ...]
) -> list[PooledSample]:
    """Attach fixed-width effective-date news vectors to causal input dates."""

    if tuple(feature_cols) != panel.feature_cols:
        raise ValueError("feature_cols must match the panel's stable sorted feature order")
    attached: list[PooledSample] = []
    matched = 0
    eligible_panel_keys = 0
    for sample in samples:
        if len(sample.input_dates) != len(sample.x_price_raw):
            raise ValueError("samples must retain input_dates for causal news alignment")
        news = np.zeros((len(sample.input_dates), len(panel.feature_cols)), dtype=np.float32)
        mask = np.zeros(len(sample.input_dates), dtype=np.int8)
        for index, date in enumerate(sample.input_dates):
            key = (sample.key.ticker, date)
            vector = panel.values.get(key)
            if key in panel.values:
                eligible_panel_keys += 1
            if vector is not None:
                news[index] = vector
                mask[index] = 1
                matched += 1
        attached.append(
            PooledSample(sample.key, sample.x_price_raw, news, mask, sample.y_raw,
                         sample.y_model_raw, sample.y_eval_raw, sample.input_dates)
        )
    if samples and eligible_panel_keys and not matched:
        raise RuntimeError("news panel keys were eligible but lookup matched none")
    return attached


def _validate_news_provenance(
    provenance: Mapping[str, object], eligible_train_cutoff: str | Mapping[str, str] | None
) -> None:
    if eligible_train_cutoff is None:
        return
    if not provenance:
        raise ValueError("learned news provenance is required for a training cutoff")
    fit_end = provenance.get("fit_period_end")
    if fit_end is None and isinstance(provenance.get("news_pca"), Mapping):
        fit_end = provenance["news_pca"].get("fit_period_end")
    if fit_end is None:
        raise ValueError("learned news provenance must include fit_period_end")
    cutoff_values = (
        eligible_train_cutoff.values()
        if isinstance(eligible_train_cutoff, Mapping)
        else [eligible_train_cutoff]
    )
    cutoff = min(pd.Timestamp(value).normalize() for value in cutoff_values)
    if pd.Timestamp(fit_end).normalize() > cutoff:
        raise ValueError("learned news fit period exceeds eligible pooled training cutoff")


def _canonical_array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(
        _stable_json({"dtype": array.dtype.str, "shape": array.shape}) + array.tobytes(order="C")
    ).hexdigest()


def _eligibility_target_hash(samples: tuple[PooledSample, ...]) -> str:
    return _stable_hash([
        (sample.key.ticker_id, sample.key.ticker, sample.key.target_date, sample.input_dates,
         sample.y_raw if sample.y_eval_raw is None else sample.y_eval_raw)
        for sample in samples
    ])


def _tensor_hash(samples: tuple[PooledSample, ...], name: str) -> str:
    return _stable_hash([_canonical_array_hash(getattr(sample, name)) for sample in samples])


def _news_tensor_hash(samples: tuple[PooledSample, ...]) -> str:
    return _stable_hash([
        (_canonical_array_hash(sample.x_news), _canonical_array_hash(sample.news_mask)) for sample in samples
    ])


def _stable_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
