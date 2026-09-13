"""Causal Diebold-Yilmaz (2012) volatility-spillover feature builder.

At each date ``t`` the spillover scalars are estimated from a trailing ``DY_WINDOW``-day VAR fit on the
sector-mean log-variance panel, using rows dated ``<= t`` ONLY -- so the feature is a function of past and
present volatility alone (no look-ahead, unlike the reference complex-network paper which estimated its
spillover graph inside the test period). See ``../design/design.md`` for the VAR + generalized-FEVD math.

Public API:
  ``sector_logvar_panel(frames, min_stocks)`` -> wide [date x sector] log-mean-variance panel.
  ``gfevd(ma, sigma)``                        -> row-normalised generalized FEVD matrix (Pesaran-Shin 1998).
  ``spillover_scalars(theta)``                -> (total spillover index S, net directional vector).
  ``spillover_from_window(win, ...)``         -> (S, net Series) for one trailing window (NaN if degenerate).
  ``rolling_spillover(panel, ...)``           -> causal (net_df, total_series, diag), ffilled to every date.
  ``merge_spillover(frames, net_df, total)``  -> frames with ``net_spillover`` / ``total_spillover`` columns.
  ``build_spillover(frames)``                 -> convenience wiring all of the above from ``config``.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import config  # noqa: E402
import full_matrix as FM  # noqa: E402

FL = FM.FL


def sector_logvar_panel(frames, min_stocks):
    """Wide log-variance panel: one column per sector with ``>= min_stocks`` constituents, value at
    (date, sector) = ``log(max(mean_i parkinson_variance_i, FL))`` over that sector's stocks present that day."""
    sec_count = Counter(d["sector"].iloc[0] for d in frames.values() if len(d))
    keep = {s for s, c in sec_count.items() if c >= min_stocks and s != -1}
    parts = []
    for d in frames.values():
        if len(d) and d["sector"].iloc[0] in keep:
            parts.append(pd.DataFrame({"date": d["date"].to_numpy(), "sector": d["sector"].iloc[0],
                                       "pv": d["parkinson_variance"].to_numpy(float)}))
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts, ignore_index=True)
    mean_pv = long.groupby(["date", "sector"])["pv"].mean().unstack("sector").sort_index()
    return np.log(np.maximum(mean_pv, FL))


def gfevd(ma, sigma):
    """Row-normalised generalized forecast-error variance decomposition (Pesaran-Shin 1998; Diebold-Yilmaz
    2012). ``ma`` is the VAR moving-average array ``[H, K, K]`` (``ma[0]=I``); ``sigma`` the ``[K, K]``
    residual covariance. Returns ``theta_tilde[i, j]`` = share of series i's H-step forecast-error variance
    attributable to shocks in series j, rows summing to 1."""
    sig = np.asarray(sigma, dtype=float)
    sig_jj = np.diag(sig)
    denom = np.zeros(sig.shape[0])
    num = np.zeros_like(sig)
    for a in ma:
        denom += np.diag(a @ sig @ a.T)
        num += (a @ sig) ** 2
    theta = (num / sig_jj[None, :]) / denom[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def spillover_scalars(theta):
    """From the row-normalised GFEVD ``theta`` return (total spillover index ``S``, net directional vector).
    ``S = 100*(1 - trace/K)``; ``NET_j = 100/K * (TO_j - FROM_j)``, positive => net volatility transmitter."""
    k = theta.shape[0]
    diag = np.diag(theta)
    to_j = theta.sum(axis=0) - diag              # column j off-diagonal: j -> others
    from_i = theta.sum(axis=1) - diag            # row i off-diagonal: others -> i
    net = (to_j - from_i) * (100.0 / k)
    s = 100.0 * (theta.sum() - diag.sum()) / k
    return float(s), net


def spillover_from_window(win, lag, gfevd_h, min_sectors, min_std, min_obs):
    """Fit a ``VAR(lag)`` on one trailing window and return ``(S, net_series)`` indexed by ``win.columns``.
    Returns ``(nan, all-nan)`` when the window is too short, has fewer than ``min_sectors`` non-degenerate
    columns, or the VAR is ill-conditioned (caught) -- the caller forward-fills such gaps."""
    nan_net = pd.Series(np.nan, index=win.columns)
    # drop any date with a missing sector value: the VAR is then fit on the retained (possibly non-contiguous)
    # window rows -- a modeling approximation on a thin panel, NOT a leakage path (all rows are still <= t).
    w = win.dropna(axis=0, how="any")
    if len(w) < min_obs:
        return float("nan"), nan_net
    keep = w.columns[w.std(axis=0) > min_std].tolist()
    if len(keep) < min_sectors:
        return float("nan"), nan_net
    wk = w[keep]
    try:
        res = VAR(wk.to_numpy(float)).fit(lag)
        ma = res.ma_rep(maxn=gfevd_h - 1)                 # [gfevd_h, K, K], ma[0] = I
        theta = gfevd(ma, res.sigma_u)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan"), nan_net
    s, net = spillover_scalars(theta)
    net_series = nan_net.copy()
    net_series.loc[keep] = net
    return s, net_series


def rolling_spillover(panel, window, step, lag, gfevd_h, min_sectors, win_min_frac, min_std):
    """Causal rolling spillover over ``panel`` (index=date, columns=sectors). Every ``step``-th date is an
    anchor scored from its trailing ``window`` rows (dates ``<= anchor``); results are reindexed onto every
    date and forward-filled. Returns ``(net_df[date x sector], total[date], diag)``."""
    dates = panel.index
    min_obs = int(window * win_min_frac)
    anchors = range(0, len(dates), step)
    net_rows, tot_rows = {}, {}
    n_failed = 0
    for pos in anchors:
        win = panel.iloc[max(0, pos - window + 1):pos + 1]
        s, net = spillover_from_window(win, lag, gfevd_h, min_sectors, min_std, min_obs)
        net_rows[dates[pos]] = net
        tot_rows[dates[pos]] = s
        if np.isnan(s):
            n_failed += 1
    net_df = pd.DataFrame(net_rows).T.reindex(columns=panel.columns).reindex(dates).ffill()
    total = pd.Series(tot_rows).reindex(dates).ffill()
    diag = {"n_sectors": int(panel.shape[1]), "n_anchor_windows": len(net_rows), "n_failed_windows": n_failed}
    return net_df, total, diag


def merge_spillover(frames, net_df, total):
    """Broadcast the causal spillover onto every ticker frame: ``net_spillover`` by (sector, date) and
    ``total_spillover`` by date. A ticker whose sector is not a VAR series gets NaN net (GBM handles NaN)."""
    out = {}
    for tk, d in frames.items():
        e = d.copy()
        sec = e["sector"].iloc[0] if len(e) else None
        e[config.FEAT_NET] = e["date"].map(net_df[sec]) if sec in net_df.columns else np.nan
        e[config.FEAT_TOTAL] = e["date"].map(total)
        out[tk] = e
    return out


def build_spillover(frames):
    """Wire the full causal feature from ``config`` constants. Returns ``(merged_frames, diag)``."""
    panel = sector_logvar_panel(frames, config.DY_SECTOR_MIN_STOCKS)
    net_df, total, diag = rolling_spillover(
        panel, config.DY_WINDOW, config.DY_STEP, config.DY_VAR_LAG, config.DY_GFEVD_H,
        config.DY_MIN_SECTORS, config.DY_WIN_MIN_FRAC, config.DY_VAR_MIN_STD)
    return merge_spillover(frames, net_df, total), diag
