"""Data utilities for the SOICT HAR-LSTM-GAT submission (self-contained).

Adapted from ``scripts/sp500_crossmarket/run_sp500_crossmarket.py`` (build_ticker_features /
make_windows / per-ticker scaling / chronological split), generalized to:
  - a configurable ``lookback`` (was a hard-coded SEQ=22),
  - an 80/10/10 chronological split (was 70/15/15),
  - a caller-supplied list of processed CSV files (data_root is chosen by the caller).

Processed CSV schema (one file per ticker): columns ``date, parkinson_variance``.
The target is Parkinson VARIANCE at ``t + horizon`` (point forecast). No leakage: windows stay
within their split and per-ticker scalers are fit on TRAIN rows only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import pipeline_config as pc

# --- rolling-window / drop thresholds (single source of truth = pipeline_config) ---
FIRST_VALID = pc.FIRST_VALID          # monthly HAR feature (22-obs rolling) is first valid at index 21
WEEKLY_WIN = pc.HAR_WEEKLY_WINDOW     # weekly HAR rolling window
MONTHLY_WIN = pc.HAR_MONTHLY_WINDOW   # monthly HAR rolling window
MIN_ROWS = pc.MIN_ROWS                # drop a ticker with fewer than this many rows
MIN_ANCHORS = pc.MIN_ANCHORS          # drop a ticker with fewer than this many valid anchors
MIN_TRAIN = pc.MIN_TRAIN              # drop if TRAIN split smaller than this
MIN_VAL = pc.MIN_VAL                  # drop if VAL split smaller than this
MIN_TEST = pc.MIN_TEST                # drop if TEST split smaller than this
_EPS = pc.SCALER_EPS                  # scaler std floor (avoid divide-by-zero)


def har_features(pk: np.ndarray) -> np.ndarray:
    """HAR feature matrix [N, 3] = [daily, weekly, monthly].

    daily   = pk(t)
    weekly  = rolling(5).mean  (NaN before index 4)
    monthly = rolling(22).mean (NaN before index 21)
    """
    pk = np.asarray(pk, dtype=np.float64)
    s = pd.Series(pk)
    daily = pk
    weekly = s.rolling(WEEKLY_WIN, min_periods=WEEKLY_WIN).mean().to_numpy()
    monthly = s.rolling(MONTHLY_WIN, min_periods=MONTHLY_WIN).mean().to_numpy()
    return np.stack([daily, weekly, monthly], axis=1)


def make_windows(pk: np.ndarray, lookback: int, horizon: int) -> np.ndarray:
    """Valid anchor indices ``t`` for a given lookback/horizon.

    A ``t`` is valid when the window ``[t-lookback+1 .. t]`` is entirely monthly-valid
    (earliest index >= FIRST_VALID) AND the target ``pk[t+horizon]`` exists.
    """
    n = len(pk)
    start = FIRST_VALID + (lookback - 1)  # earliest t whose window start is monthly-valid
    if start >= n - horizon:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, n - horizon, dtype=np.int64)


def per_stock_split(
    anchors: np.ndarray, train_frac: float = pc.TRAIN_FRAC, val_frac: float = pc.VAL_FRAC
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological, contiguous, non-overlapping split (train before val before test)."""
    anchors = np.asarray(anchors)
    n = len(anchors)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    return anchors[:i_tr], anchors[i_tr:i_va], anchors[i_va:]


class TickerScaler:
    """Per-ticker StandardScaler for HAR features and the target, fit on TRAIN rows only."""

    def __init__(self) -> None:
        self.f_mean: np.ndarray | None = None
        self.f_std: np.ndarray | None = None
        self.t_mean: float | None = None
        self.t_std: float | None = None

    def fit_features(self, feat_rows: np.ndarray) -> "TickerScaler":
        feat_rows = np.asarray(feat_rows, dtype=np.float64)
        self.f_mean = feat_rows.mean(0)
        self.f_std = feat_rows.std(0) + _EPS
        return self

    def fit_target(self, y: np.ndarray) -> "TickerScaler":
        y = np.asarray(y, dtype=np.float64)
        self.t_mean = float(y.mean())
        self.t_std = float(y.std() + _EPS)
        return self

    def transform_windows(self, windows: np.ndarray) -> np.ndarray:
        """Normalize a [M, seq, 3] window tensor with the fitted feature stats (no refit)."""
        if self.f_mean is None or self.f_std is None:
            raise RuntimeError("transform_windows called before fit_features")
        return (np.asarray(windows) - self.f_mean) / self.f_std


def _load_ticker(path: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (ticker, pk[N], feats[N,3] raw HAR, dates[N] as 'YYYY-MM-DD') or None if too short."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    pk = df["parkinson_variance"].to_numpy(dtype=np.float64)
    if len(pk) < MIN_ROWS:
        return None
    feats = har_features(pk)
    dates = df["date"].dt.strftime("%Y-%m-%d").to_numpy()
    ticker = Path(path).name.replace("_processed.csv", "")
    return ticker, pk, feats, dates


def _windows_for(feats: np.ndarray, anchors: np.ndarray, lookback: int) -> np.ndarray:
    """Gather raw [M, lookback, 3] feature windows ending at each anchor (inclusive)."""
    idx = anchors[:, None] - (lookback - 1) + np.arange(lookback)[None, :]  # [M, lookback]
    return feats[idx]  # [M, lookback, 3]


def build_pooled(
    files: list[str],
    lookback: int,
    horizon: int,
    train_frac: float = pc.TRAIN_FRAC,
    val_frac: float = pc.VAL_FRAC,
) -> dict:
    """Build a pooled dataset across tickers with per-ticker TRAIN-fit scalers.

    Drop policy (documented thresholds): a ticker is skipped when it has < MIN_ROWS (200) rows,
    < MIN_ANCHORS (60) valid anchors, or splits smaller than MIN_TRAIN/MIN_VAL/MIN_TEST (30/5/5).
    Each window is normalized by its OWN ticker's feature scaler; targets by that ticker's target
    scaler. Test tensors keep raw targets (eval on physical scale) plus ticker ids and dates.
    """
    X_tr, y_tr_norm = [], []
    X_va, y_va_norm, y_va_raw = [], [], []
    X_te, y_te_raw, te_ticker_ids, te_target_dates = [], [], [], []
    scalers: dict[int, tuple[float, float]] = {}
    ticker_order: list[str] = []

    for path in files:
        loaded = _load_ticker(path)
        if loaded is None:
            continue
        ticker, pk, feats, dates = loaded
        anchors = make_windows(pk, lookback, horizon)
        if len(anchors) < MIN_ANCHORS:
            continue
        a_tr, a_va, a_te = per_stock_split(anchors, train_frac, val_frac)
        if len(a_tr) < MIN_TRAIN or len(a_va) < MIN_VAL or len(a_te) < MIN_TEST:
            continue

        tk_id = len(ticker_order)
        ticker_order.append(ticker)

        sc = TickerScaler()
        sc.fit_features(feats[a_tr])                 # TRAIN feature rows only
        sc.fit_target(pk[a_tr + horizon])            # TRAIN targets only
        scalers[tk_id] = (sc.t_mean, sc.t_std)

        def norm_win(a: np.ndarray) -> np.ndarray:
            return sc.transform_windows(_windows_for(feats, a, lookback)).astype(np.float32)

        X_tr.append(norm_win(a_tr))
        y_tr_norm.append(((pk[a_tr + horizon] - sc.t_mean) / sc.t_std).astype(np.float32))

        X_va.append(norm_win(a_va))
        y_va_norm.append(((pk[a_va + horizon] - sc.t_mean) / sc.t_std).astype(np.float32))
        y_va_raw.append(pk[a_va + horizon].astype(np.float64))

        X_te.append(norm_win(a_te))
        y_te_raw.append(pk[a_te + horizon].astype(np.float64))
        te_ticker_ids.append(np.full(len(a_te), tk_id, dtype=np.int64))
        te_target_dates.extend(dates[a_te + horizon].tolist())

    if not ticker_order:
        raise ValueError("build_pooled: no ticker survived the drop thresholds")

    return {
        "X_tr": np.concatenate(X_tr),
        "y_tr_norm": np.concatenate(y_tr_norm),
        "X_va": np.concatenate(X_va),
        "y_va_norm": np.concatenate(y_va_norm),
        "y_va_raw": np.concatenate(y_va_raw),
        "X_te": np.concatenate(X_te),
        "y_te_raw": np.concatenate(y_te_raw),
        "te_ticker_ids": np.concatenate(te_ticker_ids),
        "te_target_dates": te_target_dates,
        "scalers": scalers,
        "ticker_order": ticker_order,
    }
