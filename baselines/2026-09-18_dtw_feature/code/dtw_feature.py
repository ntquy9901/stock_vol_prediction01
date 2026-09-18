"""Strictly-causal DTW self-similarity feature for the champion own-history gamma-GBM.

The transferable germ of Nakagawa & Yoshida (2022) "Time-series gradient boosting tree": DTW time-series
*shape* matching, adapted here as a CAUSAL FEATURE (not a new tree split criterion). For each stock-day t
we compute a Sakoe-Chiba banded DTW distance between the stock's trailing window of standardized
log-variance and a small set of reference templates built from that stock's OWN further-past (train)
history. The distance(s) are appended to the OWN-8 (+earnings) feature set.

All templates + scaling come from TRAIN rows only; every query window uses indices <= the row's own
position. `dtaidistance`/`tslearn` are not installed in this environment, so the banded DTW is a small
vectorized numpy DP (batched over the query axis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_BIG = 1e18   # DP "infinity" sentinel for out-of-band / unreachable cells (not a tunable — a numeric guard)


def banded_dtw_batch(Q: np.ndarray, template: np.ndarray, band: int) -> np.ndarray:
    """Sakoe-Chiba banded DTW distance between a batch of query windows and ONE template.

    ``Q`` is ``[B, W]`` (B query windows of length W), ``template`` is ``[W]``. Local cost is squared
    difference; the DP is vectorized over the batch axis B. ``band`` is the warping-path radius (cells
    with ``|i-j| > band`` are forbidden). Returns ``[B]`` distances = ``sqrt`` of the accumulated cost.
    Identical query and template -> 0.
    """
    Q = np.asarray(Q, dtype=float)
    t = np.asarray(template, dtype=float)
    if Q.ndim != 2 or t.ndim != 1 or Q.shape[1] != t.shape[0]:
        raise ValueError(f"Q must be [B, W] and template [W]; got {Q.shape} and {t.shape}")
    b, w = Q.shape
    cost = (Q[:, :, None] - t[None, None, :]) ** 2          # [B, W, W] pairwise local cost
    d = np.full((b, w + 1, w + 1), _BIG)
    d[:, 0, 0] = 0.0
    for i in range(1, w + 1):
        jlo, jhi = max(1, i - band), min(w, i + band)
        for j in range(jlo, jhi + 1):
            prev = np.minimum(np.minimum(d[:, i - 1, j], d[:, i, j - 1]), d[:, i - 1, j - 1])
            d[:, i, j] = cost[:, i - 1, j - 1] + prev
    return np.sqrt(d[:, w, w])


def _windows(z: np.ndarray, end_pos: np.ndarray, w: int) -> np.ndarray:
    """Trailing windows of length ``w`` ending at each position in ``end_pos`` (past-only).

    Index ``end_pos - w + 1 .. end_pos`` is clamped to ``[0, len-1]`` so early rows front-pad with the
    first available value rather than reaching before the series start (still causal)."""
    end_pos = np.asarray(end_pos, dtype=np.int64)
    idx = end_pos[:, None] + np.arange(-w + 1, 1)[None, :]
    idx = np.clip(idx, 0, len(z) - 1)
    return z[idx]


def build_templates(train_windows: np.ndarray, bands) -> np.ndarray:
    """Per-band centroid templates from standardized train windows.

    Windows are ranked by their mean level; each ``(lo, hi)`` quantile band contributes one template =
    the mean window over the ranked windows in that band. Returns ``[n_bands, W]``."""
    tw = np.asarray(train_windows, dtype=float)
    n = len(tw)
    order = np.argsort(tw.mean(axis=1))
    templates = []
    for lo, hi in bands:
        i0, i1 = int(np.floor(lo * n)), int(np.ceil(hi * n))
        sel = order[i0:i1] if i1 > i0 else order[i0:i0 + 1]
        templates.append(tw[sel].mean(axis=0))
    return np.asarray(templates, dtype=float)


def prepare_series(frames: dict) -> dict:
    """Per-ticker ``(dates[datetime64], logpk[float])`` arrays from the loaded frames (sorted by date).

    Uses the full contiguous per-ticker series (NOT the dropna'd panel) so trailing windows are built
    over uninterrupted history."""
    out = {}
    for tk, d in frames.items():
        dd = d.sort_values("date")
        out[tk] = (dd["date"].to_numpy(), dd["logpk"].to_numpy(dtype=float))
    return out


def fold_features(series: dict, combo: pd.DataFrame, boundary, cfg) -> pd.DataFrame:
    """Real + placebo DTW distance columns for every row of ``combo`` (aligned to ``combo.index``).

    ``boundary`` is the fold train cutoff (``ts - embargo``): templates + per-ticker (mu, sd) scaling use
    only rows with ``date < boundary`` (causal). The real query window ends at the row's own date; the
    placebo window ends ``cfg.DTW_PLACEBO_SHIFT`` business days earlier. Tickers with fewer than
    ``cfg.DTW_MIN_TRAIN_WINDOWS`` usable train windows yield NaN features (HGBR handles NaN natively; no
    silent zero-fill)."""
    w, band, bands = cfg.DTW_WINDOW, cfg.DTW_BAND, cfg.DTW_BANDS
    shift, min_tw = cfg.DTW_PLACEBO_SHIFT, cfg.DTW_MIN_TRAIN_WINDOWS
    n_t = len(bands)
    real_cols = [f"dtw{i}" for i in range(n_t)]
    plac_cols = [f"dtwp{i}" for i in range(n_t)]
    real = np.full((len(combo), n_t), np.nan)
    plac = np.full((len(combo), n_t), np.nan)
    row_of_label = {lbl: i for i, lbl in enumerate(combo.index)}
    bnd = np.datetime64(pd.Timestamp(boundary))
    for tk, g in combo.groupby("ticker", sort=False):
        if tk not in series:
            continue
        dts, lpk = series[tk]
        train_mask = dts < bnd
        if int(train_mask.sum()) < w:
            continue
        mu = float(lpk[train_mask].mean())
        sd = float(lpk[train_mask].std())
        z = (lpk - mu) / (sd if sd > 0 else 1.0)
        train_end = np.where(train_mask)[0]
        tw = _windows(z, train_end, w)
        if len(tw) < min_tw:
            continue
        templates = build_templates(tw, bands)
        pos = np.searchsorted(dts, g["date"].to_numpy())      # each combo row's position in the series
        q = _windows(z, pos, w)
        qp = _windows(z, np.clip(pos - shift, 0, None), w)
        rows = np.array([row_of_label[lbl] for lbl in g.index])
        for ti in range(n_t):
            real[rows, ti] = banded_dtw_batch(q, templates[ti], band)
            plac[rows, ti] = banded_dtw_batch(qp, templates[ti], band)
    out = pd.DataFrame(np.hstack([real, plac]), columns=real_cols + plac_cols, index=combo.index)
    return out


def feature_names(cfg) -> tuple[list, list]:
    """(real DTW cols, placebo DTW cols) for the configured number of templates."""
    n_t = len(cfg.DTW_BANDS)
    return [f"dtw{i}" for i in range(n_t)], [f"dtwp{i}" for i in range(n_t)]
