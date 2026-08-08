"""Train-fitted per-ticker preprocessing for the pooled pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_EPSILON = 1e-8


@dataclass(frozen=True)
class ArrayStandardizer:
    """A small serializable standardizer with the project's zero-variance policy."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "ArrayStandardizer":
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError("scaler fit values must be a non-empty finite two-dimensional array")
        std = array.std(axis=0)
        return ArrayStandardizer(array.mean(axis=0), np.where(std < _EPSILON, 1.0, std))

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return (np.asarray(values, dtype=float) - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(values, dtype=float) * self.std + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        self._check_fitted()
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: dict[str, list[float]]) -> "ArrayStandardizer":
        return cls(np.asarray(value["mean"], dtype=float), np.asarray(value["std"], dtype=float))

    def _check_fitted(self) -> None:
        if self.mean is None or self.std is None:
            raise ValueError("scaler must be fitted before transformation")


@dataclass(frozen=True)
class TickerPreprocessor:
    """All per-ticker fitted values, derived exclusively from raw train rows."""

    feature_order: tuple[str, ...]
    target_column: str
    lower_bound: float
    upper_bound: float
    feature_scaler: ArrayStandardizer
    target_scaler: ArrayStandardizer

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        train_features: list[str],
        train_targets: str,
    ) -> "TickerPreprocessor":
        if len(train_features) != 1 or train_features[0] != train_targets:
            raise ValueError("the pilot expects one raw volatility feature identical to the target")
        if train_targets not in train_frame:
            raise ValueError(f"missing target column: {train_targets}")
        raw_target = train_frame[train_targets].to_numpy(dtype=float)
        if raw_target.size == 0 or not np.isfinite(raw_target).all():
            raise ValueError("train target values must be non-empty and finite")
        mean = float(raw_target.mean())
        std = float(raw_target.std())
        lower_bound, upper_bound = mean - 3 * std, mean + 3 * std
        clipped = np.clip(raw_target, lower_bound, upper_bound)
        features = _har_features(clipped)
        valid_rows = np.isfinite(features).all(axis=1)
        return cls(
            feature_order=(train_targets, "har_weekly", "har_monthly"),
            target_column=train_targets,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            feature_scaler=ArrayStandardizer().fit(features[valid_rows]),
            target_scaler=ArrayStandardizer().fit(clipped[valid_rows]),
        )

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply fixed train parameters then create split-local HAR and normalized inputs."""

        if self.target_column not in frame:
            raise ValueError(f"missing target column: {self.target_column}")
        result = frame.copy()
        y_eval = result[self.target_column].to_numpy(dtype=float)
        if not np.isfinite(y_eval).all():
            raise ValueError("split target values must be finite")
        y_model = np.clip(y_eval, self.lower_bound, self.upper_bound)
        features = _har_features(y_model)
        valid_rows = np.isfinite(features).all(axis=1)
        result = result.loc[valid_rows].copy()
        normalized = self.feature_scaler.transform(features[valid_rows])
        y_model = y_model[valid_rows]
        y_eval = y_eval[valid_rows]
        result["y_model_raw"] = y_model
        result["y_eval_raw"] = y_eval
        for index, name in enumerate(self.feature_order):
            result[f"feature_{name}"] = normalized[:, index]
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_order": list(self.feature_order),
            "target_column": self.target_column,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "feature_scaler": self.feature_scaler.to_dict(),
            "target_scaler": self.target_scaler.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TickerPreprocessor":
        return cls(
            feature_order=tuple(value["feature_order"]),  # type: ignore[arg-type]
            target_column=str(value["target_column"]),
            lower_bound=float(value["lower_bound"]),
            upper_bound=float(value["upper_bound"]),
            feature_scaler=ArrayStandardizer.from_dict(value["feature_scaler"]),  # type: ignore[arg-type]
            target_scaler=ArrayStandardizer.from_dict(value["target_scaler"]),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PreprocessorStore:
    """Explicit ticker-ID dispatcher for per-ticker fitted preprocessing."""

    preprocessors: dict[int, TickerPreprocessor]

    def transform_features(self, ticker_id: int, values: np.ndarray) -> np.ndarray:
        return self.get(ticker_id).feature_scaler.transform(values)

    def get(self, ticker_id: int) -> TickerPreprocessor:
        """Return one train-fitted ticker preprocessor or fail before transformation."""

        try:
            return self.preprocessors[ticker_id]
        except KeyError as error:
            raise ValueError(f"missing preprocessor for ticker_id {ticker_id}") from error

    def inverse_targets(self, ticker_ids: np.ndarray, y_norm: np.ndarray) -> np.ndarray:
        ids = np.asarray(ticker_ids)
        normalized = np.asarray(y_norm, dtype=float)
        if ids.shape != normalized.shape:
            raise ValueError("ticker_ids and y_norm must have matching shapes")
        restored = [
            self.get(int(ticker_id)).target_scaler.inverse_transform(np.array([value]))[0]
            for ticker_id, value in zip(ids, normalized, strict=True)
        ]
        return np.asarray(restored, dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            str(ticker_id): preprocessor.to_dict()
            for ticker_id, preprocessor in sorted(self.preprocessors.items())
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PreprocessorStore":
        return cls(
            {
                int(ticker_id): TickerPreprocessor.from_dict(preprocessor)  # type: ignore[arg-type]
                for ticker_id, preprocessor in value.items()
            }
        )

def _har_features(clipped_values: np.ndarray) -> np.ndarray:
    series = pd.Series(clipped_values, dtype=float)
    return np.column_stack(
        (
            clipped_values,
            series.rolling(5).mean().to_numpy(),
            series.rolling(22).mean().to_numpy(),
        )
    )
