"""Causal feature construction for the XGBoost residual-ratio study.

Every feature at forecast origin ``t`` uses information available no later than ``t``:

* **Stock** features are computed per ticker on its OWN trading series (then reindexed to the union
  calendar) -- the same per-ticker convention as the delivered ``har_weekly``/``har_monthly`` -- so a
  lag/rolling window never bleeds across a ticker's non-trading gaps. All rolling windows end at ``t``
  (they may include ``pk_t``, which is realised and known at the close of day ``t``); none look ahead.
* **Market** features are cross-sectional over the valid nodes at date ``t`` only.
* **Graph** features aggregate neighbours' same-day values through a TRAIN-only adjacency (built by the
  caller with ``last_row = last_train_anchor + horizon``); ``A[target j, source i]`` so
  ``neighbor_x[j,t] = sum_i A[j,i] * x_i(t)``.

The builders return dense ``[T, N, F]`` tensors + column-name and feature-group lists; NaN on early rows
(insufficient history) is left as NaN for XGBoost's native ``hist`` NaN routing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import xgb_config as C


def read_aligned_columns(files, panel, cols):
    """Read ``cols`` from each ticker CSV, aligned to ``panel.dates`` x ``panel.tickers`` -> dict of
    ``[T,N]`` float arrays (NaN off a ticker's own dates). Only the panel's kept tickers are read."""
    keep = {t: j for j, t in enumerate(panel.tickers)}
    out = {c: np.full((len(panel.dates), panel.N), np.nan) for c in cols}
    for f in files:
        tk = Path(f).stem
        if tk not in keep:
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
        for c in cols:
            if c in df.columns:
                out[c][:, keep[tk]] = df[c].reindex(panel.dates).to_numpy(dtype=float)
    return out


def _pernode(mat, fn):
    """Apply ``fn`` (a pandas Series -> ndarray/Series transform) to each column's OWN (finite) values,
    writing results back at those positions. Off-own-date positions stay NaN."""
    t, n = mat.shape
    out = np.full((t, n), np.nan)
    for j in range(n):
        col = mat[:, j]
        own = np.flatnonzero(np.isfinite(col))
        if own.size == 0:
            continue
        res = np.asarray(fn(pd.Series(col[own].astype(float))), dtype=float)
        out[own, j] = res
    return out


def _rolling_slope(mat, window):
    """Per-node rolling OLS slope of pk against time over ``window`` own-trading-days (backward-looking).

    For a window ``y_0..y_{w-1}`` against ``x = 0..w-1`` the slope is
    ``sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)`` with a constant denominator; vectorised over all windows
    via ``sliding_window_view`` (no future data enters -- the window ends at the current own-day)."""
    def fn(s):
        v = s.to_numpy(dtype=float)
        out = np.full(v.size, np.nan)
        if v.size >= window:
            w = sliding_window_view(v, window)                    # [L-w+1, w] ending at positions w-1..
            x = np.arange(window, dtype=float)
            xc = x - x.mean()
            denom = float((xc ** 2).sum())
            yc = w - w.mean(axis=1, keepdims=True)
            out[window - 1:] = (yc * xc).sum(axis=1) / denom
        return out
    return _pernode(mat, fn)


def build_sm_features(panel, ret, logrange, volz):
    """Stock + market causal features -> ``(F [T,N,Fsm], names, groups)``.

    ``ret``/``logrange`` are ``[T,N]`` aligned columns (daily_return / log_range); ``volz`` is the
    per-ticker volume z-score. ``groups[k]`` is ``"stock"`` or ``"market"`` -- the ablation switch.
    """
    pk = panel.pk                                                 # [T,N] Parkinson variance
    har_w = panel.feats[:, :, 1]
    har_m = panel.feats[:, :, 2]
    market_pk = panel.feats[:, :, 3]
    cols, names, groups = [], [], []

    def add(arr, name, group):
        cols.append(arr); names.append(name); groups.append(group)

    # --- stock (per-ticker, own series) ---
    add(pk, "pk_t", "stock")
    for k in C.PK_LAGS:
        add(_pernode(pk, lambda s, k=k: s.shift(k)), f"pk_lag{k}", "stock")
    for w in C.ROLL_MEANS:
        add(_pernode(pk, lambda s, w=w: s.rolling(w, min_periods=w).mean()), f"pk_rollmean{w}", "stock")
    add(_pernode(pk, lambda s: s.rolling(C.ROLL_STD_WINDOW, min_periods=C.ROLL_STD_WINDOW).std(ddof=1)),
        f"pk_rollstd{C.ROLL_STD_WINDOW}", "stock")
    add(_pernode(pk, lambda s: s.rolling(C.ROLL_MED_WINDOW, min_periods=C.ROLL_MED_WINDOW).median()),
        f"pk_rollmed{C.ROLL_MED_WINDOW}", "stock")
    add(_pernode(pk, lambda s: (s - s.rolling(C.ROLL_MED_WINDOW, min_periods=C.ROLL_MED_WINDOW).median())
                 .abs().rolling(C.ROLL_MED_WINDOW, min_periods=C.ROLL_MED_WINDOW).median()),
        f"pk_mad{C.ROLL_MED_WINDOW}", "stock")
    for sp in C.EWMA_SPANS:
        add(_pernode(pk, lambda s, sp=sp: s.ewm(span=sp, adjust=False).mean()), f"pk_ewma{sp}", "stock")
    rmw = _pernode(pk, lambda s: s.rolling(C.RATIO_WINDOW, min_periods=C.RATIO_WINDOW).mean())
    add(pk / (rmw + C.RATIO_EPS), f"pk_ratio{C.RATIO_WINDOW}", "stock")
    add(_rolling_slope(pk, C.SLOPE_WINDOW), f"pk_slope{C.SLOPE_WINDOW}", "stock")
    add(ret, "ret_t", "stock")
    add(np.abs(ret), "absret_t", "stock")
    add((ret < 0).astype(float) * np.where(np.isfinite(ret), 1.0, np.nan), "negret_ind", "stock")
    add(logrange, "log_range_t", "stock")
    add(har_w, "har_weekly", "stock")
    add(har_m, "har_monthly", "stock")
    add(volz, "volume_zscore", "stock")

    # --- market (cross-sectional at date t, broadcast to all nodes) ---
    def bcast(vec):
        return np.repeat(vec[:, None], panel.N, axis=1)

    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN cross-section on a no-trade date -> NaN (masked)
        cs_mean = np.nanmean(pk, axis=1)
        cs_std = np.nanstd(pk, axis=1)
        cs_med = np.nanmedian(pk, axis=1)
        cs_mad = np.nanmedian(np.abs(pk - cs_med[:, None]), axis=1)
        mkt_ret = np.nanmean(ret, axis=1)
        up = np.nansum((ret > 0).astype(float) * np.isfinite(ret), axis=1)
        down = np.nansum((ret < 0).astype(float) * np.isfinite(ret), axis=1)
        cnt = np.sum(np.isfinite(ret), axis=1).astype(float)
        cnt_safe = np.where(cnt > 0, cnt, np.nan)
        mkt_shock = np.nanmean(volz, axis=1)
        market_pk_row = np.nanmean(market_pk, axis=1)   # market_pk is shared across nodes; nanmean recovers it, NaN-safe
    add(bcast(market_pk_row), "market_pk", "market")
    add(bcast(cs_mean), "cs_mean_pk", "market")
    add(bcast(cs_std), "cs_std_pk", "market")
    add(bcast(cs_mad), "cs_mad_pk", "market")
    add(bcast(mkt_ret), "market_ret", "market")
    add(bcast(up / cnt_safe), "frac_up", "market")
    add(bcast(down / cnt_safe), "frac_down", "market")
    add(bcast(mkt_shock), "market_vol_shock", "market")

    F = np.stack(cols, axis=2).astype(np.float32)                 # [T,N,Fsm]
    return F, names, groups


def build_graph_features(panel, ret, adj):
    """Neighbour-aggregated causal features through a TRAIN-only adjacency -> ``(G [T,N,3], names)``.

    ``adj[j,i]`` weights source ``i`` into target ``j`` (``masked_rich`` orientation), so
    ``neighbor_x[j,t] = sum_i adj[j,i] * x_i(t)`` over the SAME day ``t`` (all inputs known at ``t``).
    NaN inputs are zeroed inside the sum (a non-trading neighbour contributes nothing)."""
    pk = np.nan_to_num(panel.pk)                                  # [T,N]
    volz = np.nan_to_num(panel.feats[:, :, 4])
    r = np.nan_to_num(ret)
    npk = pk @ adj.T                                              # [T,N]  (sum_i adj[j,i]*pk_i)
    nshock = volz @ adj.T
    nret = r @ adj.T
    G = np.stack([npk, nshock, nret], axis=2).astype(np.float32)
    return G, ["neighbor_pk", "neighbor_shock", "neighbor_return"]
