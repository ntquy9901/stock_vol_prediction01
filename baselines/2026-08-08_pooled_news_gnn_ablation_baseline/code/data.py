"""Validated raw price records and chronological per-ticker splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from datetime import time
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
        price = np.asarray(self.x_price_raw)
        news = np.asarray(self.x_news)
        mask = np.asarray(self.news_mask)
        if price.ndim != 2 or news.ndim != 2:
            raise ValueError("x_price_raw and x_news must be 2-D")
        if price.shape[0] != news.shape[0]:
            raise ValueError("price and news sequences must have equal time dimensions")
        if mask.ndim != 1 or mask.shape[0] != price.shape[0] or not np.isin(mask, (0, 1)).all():
            raise ValueError("news_mask must be a binary 1-D sequence matching the time dimension")
        for name, value in (("y_raw", self.y_raw), ("y_model_raw", self.y_model_raw),
                            ("y_eval_raw", self.y_eval_raw)):
            if value is not None:
                try:
                    if not np.isfinite(float(value)):
                        raise ValueError(f"{name} must be finite")
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{name} must be a finite scalar") from error
        object.__setattr__(self, "x_price_raw", _readonly_array(price, dtype=np.float32))
        object.__setattr__(self, "x_news", _readonly_array(news, dtype=np.float32))
        object.__setattr__(self, "news_mask", _readonly_array(mask, dtype=np.int8))
        normalized_dates = tuple(_normalize_local_date(date) for date in self.input_dates)
        if normalized_dates and len(normalized_dates) != len(self.x_price_raw):
            raise ValueError("input_dates must match the price sequence length")
        object.__setattr__(self, "input_dates", normalized_dates)


@dataclass(frozen=True)
class NewsPanel:
    """Validated effective-trading-date news vectors keyed by ticker and date."""

    values: Mapping[tuple[str, str], np.ndarray]
    feature_cols: tuple[str, ...]
    provenance: Mapping[str, object]
    eligible_keys: frozenset[tuple[str, str]] = frozenset()

    def __post_init__(self) -> None:
        values = {}
        width = len(self.feature_cols)
        for (ticker, date), vector in self.values.items():
            array = np.asarray(vector)
            if array.ndim != 1 or array.shape[0] != width:
                raise ValueError("news vectors must be 1-D and match feature_cols")
            values[(str(ticker), _normalize_local_date(date))] = _readonly_array(array, dtype=np.float32)
        object.__setattr__(self, "values", MappingProxyType(values))
        object.__setattr__(self, "feature_cols", tuple(self.feature_cols))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        keys = self.eligible_keys or frozenset(values)
        object.__setattr__(self, "eligible_keys", frozenset(keys))

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


@dataclass(frozen=True)
class GraphNode:
    """One fixed-vocabulary node inside a graph snapshot."""

    ticker_id: int
    ticker: str
    split: str
    y_raw: float
    y_norm: float = 0.0


@dataclass(frozen=True)
class GraphSnapshot:
    """A common-date graph input whose nodes all belong to one temporal split."""

    target_date: str
    split: str
    input_dates: tuple[str, ...]
    nodes: tuple[GraphNode, ...]
    x_price: np.ndarray
    x_news: np.ndarray
    news_mask: np.ndarray
    adjacency: np.ndarray

    def __post_init__(self) -> None:
        node_count = len(self.nodes)
        if self.split not in _SPLIT_NAMES or not node_count:
            raise ValueError("graph snapshots require nodes and a valid split")
        if any(node.split != self.split for node in self.nodes):
            raise ValueError("graph snapshot nodes must share the snapshot split")
        price = np.asarray(self.x_price, dtype=np.float32)
        news = np.asarray(self.x_news, dtype=np.float32)
        mask = np.asarray(self.news_mask, dtype=np.int8)
        adjacency = np.asarray(self.adjacency, dtype=np.float32)
        if price.ndim != 3 or price.shape[0] != node_count:
            raise ValueError("graph x_price must be [nodes, sequence, features]")
        if news.ndim != 3 or news.shape[:2] != price.shape[:2] or mask.shape != price.shape[:2]:
            raise ValueError("graph news tensors must align with price nodes and sequence")
        if adjacency.shape != (node_count, node_count) or not np.isfinite(adjacency).all():
            raise ValueError("graph adjacency must be finite and square over nodes")
        object.__setattr__(self, "x_price", _readonly_array(price, np.float32))
        object.__setattr__(self, "x_news", _readonly_array(news, np.float32))
        object.__setattr__(self, "news_mask", _readonly_array(mask, np.int8))
        object.__setattr__(self, "adjacency", _readonly_array(adjacency, np.float32))


@dataclass(frozen=True)
class GraphManifest:
    """Matched G0/G1 graph snapshots derived on a single common date axis."""

    snapshots: tuple[GraphSnapshot, ...]
    ticker_to_id: Mapping[str, int]
    train_end_date: str
    val_end_date: str
    hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        vocabulary = dict(sorted(self.ticker_to_id.items(), key=lambda item: item[1]))
        if list(vocabulary.values()) != list(range(len(vocabulary))):
            raise ValueError("graph node vocabulary must have contiguous ticker IDs")
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        object.__setattr__(self, "ticker_to_id", MappingProxyType(vocabulary))
        object.__setattr__(self, "hashes", MappingProxyType(dict(self.hashes)))

    def content_hash(self, split: str) -> str:
        if split not in _SPLIT_NAMES:
            raise ValueError(f"unknown graph split: {split}")
        return _stable_hash({"split": split, "hashes": dict(self.hashes), "snapshots": [
            _graph_snapshot_hash(snapshot) for snapshot in self.snapshots if snapshot.split == split
        ]})


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
                 _raw_target_hash_from_dict(sample))
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


def _raw_target_hash(sample: PooledSample) -> tuple[str, str, str]:
    return tuple(_canonical_scalar_hash(value) for value in (sample.y_raw, sample.y_model_raw, sample.y_eval_raw))


def _raw_target_hash_from_dict(sample: Mapping[str, object]) -> tuple[str, str, str]:
    return tuple(_canonical_scalar_hash(sample[name]) for name in ("y_raw", "y_model_raw", "y_eval_raw"))


def _canonical_scalar_hash(value: object) -> str:
    if value is None:
        return _stable_hash({"value": None})
    number = np.asarray([float(value)], dtype="<f8")
    return _canonical_array_hash(number)


def _readonly_array(values: np.ndarray, dtype: type[np.floating] | type[np.int8]) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("sample arrays must be finite")
    array.setflags(write=False)
    return array


def _normalize_local_date(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("news panel date must not be null")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
    return timestamp.strftime("%Y-%m-%d")


def effective_trading_date(news_dt: pd.Series, trading_dates: Iterable[object], close_hour: int = 15) -> pd.Series:
    """Map timestamps to the next available trading date using a strict close boundary."""

    calendar = sorted({_normalize_local_date(value) for value in trading_dates})
    output: list[pd.Timestamp] = []
    for value in news_dt:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            output.append(pd.NaT)
            continue
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
        candidate = timestamp.normalize()
        if timestamp.time() >= time(close_hour, 0):
            candidate += pd.Timedelta(days=1)
        eligible = next((pd.Timestamp(day) for day in calendar if pd.Timestamp(day) >= candidate), pd.NaT)
        output.append(eligible)
    return pd.Series(output, index=news_dt.index)


def load_effective_news_panel(
    path: Path | str,
    eligible_train_cutoff: str | Mapping[str, str] | None = None,
    tickers: Iterable[str] | None = None,
    require_provenance: bool = False,
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
        normalized["date"] = normalized["date"].map(_normalize_local_date)
    except (TypeError, ValueError) as error:
        raise ValueError("news panel dates must be parseable and non-null") from error
    if normalized.duplicated(["ticker", "date"]).any():
        raise ValueError("news panel (ticker, date) keys must be unique")
    values = normalized.loc[:, feature_cols].to_numpy(dtype=np.float32)
    missing_rows = np.isnan(values).all(axis=1)
    provenance_path = panel_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    partial_rows = ~missing_rows & np.isnan(values).any(axis=1)
    if np.isinf(values).any() or (partial_rows.any() and provenance.get("missing_value_policy") != "zero_fill"):
        raise ValueError("news panel features must be finite")
    values = np.nan_to_num(values, nan=0.0)
    _validate_news_provenance(provenance, eligible_train_cutoff, require_provenance)
    return NewsPanel(
        values={
            (str(row.ticker), str(row.date)): values[index]
            for index, row in enumerate(normalized.loc[:, ["ticker", "date"]].itertuples(index=False))
            if not missing_rows[index]
        },
        feature_cols=feature_cols,
        provenance=provenance,
        eligible_keys=frozenset(
            (str(row.ticker), str(row.date))
            for index, row in enumerate(normalized.loc[:, ["ticker", "date"]].itertuples(index=False))
            if not missing_rows[index]
        ),
    )


def attach_news(
    samples: list[PooledSample], panel: NewsPanel, feature_cols: list[str] | tuple[str, ...]
) -> list[PooledSample]:
    """Attach fixed-width effective-date news vectors to causal input dates."""

    if tuple(feature_cols) != panel.feature_cols:
        raise ValueError("feature_cols must match the panel's stable sorted feature order")
    attached: list[PooledSample] = []
    successful_keys: set[tuple[str, str]] = set()
    sample_keys: set[tuple[str, str]] = set()
    for sample in samples:
        if len(sample.input_dates) != len(sample.x_price_raw):
            raise ValueError("samples must retain input_dates for causal news alignment")
        news = np.zeros((len(sample.input_dates), len(panel.feature_cols)), dtype=np.float32)
        mask = np.zeros(len(sample.input_dates), dtype=np.int8)
        for index, date in enumerate(sample.input_dates):
            key = (sample.key.ticker, date)
            sample_keys.add(key)
            vector = panel.values.get(key)
            if vector is not None:
                news[index] = vector
                mask[index] = 1
                successful_keys.add(key)
        attached.append(
            PooledSample(sample.key, sample.x_price_raw, news, mask, sample.y_raw,
                         sample.y_model_raw, sample.y_eval_raw, sample.input_dates)
        )
    expected_keys = set(panel.eligible_keys).intersection(sample_keys)
    if expected_keys != successful_keys:
        raise RuntimeError("news coverage mismatch between eligible keys and attachments")
    return attached


def _validate_news_provenance(
    provenance: Mapping[str, object],
    eligible_train_cutoff: str | Mapping[str, str] | None,
    require_provenance: bool,
) -> None:
    if eligible_train_cutoff is None and not require_provenance:
        return
    if not provenance:
        raise ValueError("learned news provenance is required for a training cutoff")
    fit_end = provenance.get("fit_period_end")
    if fit_end is None and isinstance(provenance.get("news_pca"), Mapping):
        fit_end = provenance["news_pca"].get("fit_period_end")
    if fit_end is None:
        raise ValueError("learned news provenance must include fit_period_end")
    if eligible_train_cutoff is None:
        return
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
         _raw_target_hash(sample))
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


def _transform_full_frames(
    price_frames: Mapping[str, pd.DataFrame], preprocessors: PreprocessorStore
) -> tuple[dict[str, pd.DataFrame], tuple[str, ...]]:
    """Transform each full ticker frame with train-fitted state and intersect their dates.

    The intersection is the globally-common post-HAR trading axis shared by the graph
    manifest and the common-date pooled regime: a date survives only when every ticker
    both traded on it and had a complete HAR warm-up by then.
    """

    ordered_tickers = sorted(price_frames)
    normalized: dict[str, pd.DataFrame] = {}
    date_sets: list[set[str]] = []
    for ticker in ordered_tickers:
        frame = price_frames[ticker].copy()
        _validated_dates(frame)
        if "parkinson_volatility" not in frame:
            raise ValueError("graph price frames require parkinson_volatility")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
        if not np.isfinite(frame["parkinson_volatility"].to_numpy(dtype=float)).all():
            raise ValueError("graph price values must be finite")
        ticker_id = ordered_tickers.index(ticker)
        transformed = preprocessors.get(ticker_id).transform_frame(frame)
        normalized[ticker] = transformed
        date_sets.append(set(transformed["date"]))
    common_dates = sorted(set.intersection(*date_sets))
    if not common_dates:
        raise ValueError("graph price frames have no common dates")
    return normalized, tuple(common_dates)


def common_trading_dates(
    price_frames: Mapping[str, pd.DataFrame], preprocessors: PreprocessorStore
) -> tuple[str, ...]:
    """Return the sorted globally-common post-HAR trading dates (the graph common date axis)."""

    return _transform_full_frames(price_frames, preprocessors)[1]


def restrict_manifest_to_common_dates(
    manifest: PooledManifest, common_dates: Iterable[str]
) -> PooledManifest:
    """Keep only pooled samples whose entire window lies on globally-common trading dates.

    This is the A1 sample-set operation: the per-ticker split, train-fitted scalers/winsor
    bounds, sequence length, horizon, and any attached news are all carried through unchanged;
    only the training-sample SET is reduced to the common-date intersection.

    Windows remain per-ticker contiguous in that ticker's own trading days (the horizon gap is
    counted in per-ticker days, as in build_pooled_manifest); the common axis only decides which
    whole windows survive. It is not a re-windowing onto a shared, gap-free common-date grid.
    """

    common = frozenset(common_dates)
    filtered: dict[str, tuple[PooledSample, ...]] = {}
    for split in _SPLIT_NAMES:
        kept = tuple(
            sample
            for sample in manifest.samples[split]
            if sample.key.target_date in common and set(sample.input_dates) <= common
        )
        if not kept:
            raise ValueError(f"common-date restriction removed all {split} samples")
        filtered[split] = kept
    return PooledManifest(filtered, manifest.exclusions, manifest.ticker_to_id, manifest.preprocessing_hash)


def build_graph_manifest(
    price_frames: Mapping[str, pd.DataFrame],
    news_panel: NewsPanel,
    preprocessors: PreprocessorStore,
    seq_length: int = 22,
    horizon: int = 5,
) -> GraphManifest:
    """Build matched graph snapshots on the price-date intersection only.

    Splitting the common date axis before windows makes the graph boundary explicit:
    neither an input observation nor a target can cross from one graph split into another.
    """

    if seq_length <= 0 or horizon <= 0:
        raise ValueError("seq_length and horizon must be positive")
    if len(price_frames) < 2:
        raise ValueError("graph ablation requires at least two ticker frames")
    ordered_tickers = sorted(price_frames)
    transformed_frames, common_dates = _transform_full_frames(price_frames, preprocessors)
    normalized = {
        ticker: frame.set_index("date", drop=False) for ticker, frame in transformed_frames.items()
    }
    counts = _global_split_counts(len(common_dates))
    ticker_to_id = {ticker: index for index, ticker in enumerate(ordered_tickers)}
    snapshots: list[GraphSnapshot] = []
    start = 0
    for split, count in zip(_SPLIT_NAMES, counts, strict=True):
        split_dates = common_dates[start : start + count]
        start += count
        for offset in range(max(0, len(split_dates) - seq_length - horizon + 1)):
            input_dates = tuple(split_dates[offset : offset + seq_length])
            target_date = split_dates[offset + seq_length + horizon - 1]
            nodes = tuple(
                GraphNode(
                    ticker_to_id[ticker], ticker, split,
                    float(normalized[ticker].loc[target_date, "y_eval_raw"]),
                    float(preprocessors.get(ticker_to_id[ticker]).target_scaler.transform(
                        np.asarray([normalized[ticker].loc[target_date, "y_model_raw"]])
                    )[0]),
                )
                for ticker in ordered_tickers
            )
            price = np.stack([
                normalized[ticker].loc[
                    list(input_dates),
                    [f"feature_{name}" for name in preprocessors.get(ticker_to_id[ticker]).feature_order],
                ].to_numpy(dtype=np.float32)
                for ticker in ordered_tickers
            ])
            news, mask = _graph_news_tensors(ordered_tickers, input_dates, news_panel)
            snapshots.append(GraphSnapshot(target_date, split, input_dates, nodes, price, news, mask,
                                           _correlation_adjacency(price[:, :, 0])))
    if not snapshots:
        raise ValueError(
            "graph manifest produced no snapshots: the common date axis is too short for "
            f"seq_length={seq_length} and horizon={horizon}"
        )
    hashes = {
        "node_vocabulary": _stable_hash(ticker_to_id),
        "snapshots": _stable_hash([_graph_snapshot_hash(snapshot) for snapshot in snapshots]),
        "adjacency": _stable_hash([_canonical_array_hash(snapshot.adjacency) for snapshot in snapshots]),
        "tensors": _stable_hash([(
            _canonical_array_hash(snapshot.x_price), _canonical_array_hash(snapshot.x_news),
            _canonical_array_hash(snapshot.news_mask),
        ) for snapshot in snapshots]),
    }
    return GraphManifest(tuple(snapshots), ticker_to_id, common_dates[counts[0] - 1],
                         common_dates[counts[0] + counts[1] - 1], hashes)


def _global_split_counts(length: int) -> tuple[int, int, int]:
    counts = (int(length * 0.7), int(length * 0.15), length - int(length * 0.7) - int(length * 0.15))
    if any(count <= 0 for count in counts):
        raise ValueError("each global graph split must contain at least one date")
    return counts


def _graph_news_tensors(
    tickers: list[str], input_dates: tuple[str, ...], panel: NewsPanel
) -> tuple[np.ndarray, np.ndarray]:
    width = len(panel.feature_cols)
    news = np.zeros((len(tickers), len(input_dates), width), dtype=np.float32)
    mask = np.zeros((len(tickers), len(input_dates)), dtype=np.int8)
    for node_index, ticker in enumerate(tickers):
        for date_index, date in enumerate(input_dates):
            vector = panel.values.get((ticker, date))
            if vector is not None:
                news[node_index, date_index] = vector
                mask[node_index, date_index] = 1
    return news, mask


def _correlation_adjacency(values: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(values)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _graph_snapshot_hash(snapshot: GraphSnapshot) -> dict[str, object]:
    return {
        "target_date": snapshot.target_date, "split": snapshot.split, "input_dates": snapshot.input_dates,
        "nodes": [
            (node.ticker_id, node.ticker, node.split, _canonical_scalar_hash(node.y_raw),
             _canonical_scalar_hash(node.y_norm))
            for node in snapshot.nodes
        ],
        "price": _canonical_array_hash(snapshot.x_price), "news": _canonical_array_hash(snapshot.x_news),
        "mask": _canonical_array_hash(snapshot.news_mask), "adjacency": _canonical_array_hash(snapshot.adjacency),
    }
