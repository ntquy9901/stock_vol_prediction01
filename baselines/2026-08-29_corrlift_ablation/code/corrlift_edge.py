"""Combined linear + non-linear graph edge (Sonani, Badii & Moin 2025, arXiv:2502.15813 §3.2).

Replicates the paper's COMBINED undirected adjacency for the HNX volatility LSTM+GAT probe:

  1. LINEAR -- Pearson correlation on daily returns.
     ``r_t = (P_t - P_{t-1}) / P_{t-1}``            (their Eq. 2)
     ``rho_ij = cov(r_i, r_j) / (std(r_i) std(r_j))`` (their Eq. 3, the standard Pearson formula)
     Edge fires if ``|rho_ij| > 0.7``.
  2. NON-LINEAR -- association-rule LIFT (Apriori / market-basket).
     Item (per stock) = a "notable move": a trading day whose ``|return|`` exceeds that stock's TRAIN-median
     ``|return|`` (a per-stock notable-move indicator). A transaction = one trading day.
     Standard market-basket definitions (Agrawal & Srikant 1994; Han, Kamber & Pei, "Data Mining"):
       ``support(X) = P(X)``                       = fraction of transactions containing item set X
       ``lift(i, j) = support(i, j) / (support(i) * support(j))``
     ``lift > 1`` means i and j co-occur MORE than independence predicts. Edge fires if ``lift_ij > 1.7``
     (co-move ~70% more than chance -- the paper's threshold).

Combine: an edge is present if EITHER criterion fires; the weight is the mean of the two normalised
strengths (``|rho|`` in [0,1]; lift excess ``lift-1`` MAX-scaled by the largest fired excess -> (0,1]),
averaging only the criteria that actually fired. Undirected (symmetric); self-loop = 1.0 to match the
``WeightedGATLayer`` convention used by ``adj_vol2pk`` / ``adj_corr``.

LEAKAGE (strict -- the paper is silent; our H1 lesson): the whole graph (returns, correlations, per-stock
item thresholds, supports, lifts) is computed from TRAIN close rows ONLY (``date < cutoff_date``) and then
FROZEN for val/test -- mirroring how ``masked_rich`` fits ``adj_vol2pk`` on train rows only. The caller uses
``cutoff = D.d_va[0]`` (the first VALIDATION target date): every included close row is strictly before every
val/test target, so there is NO evaluation leakage. This boundary is marginally LOOSER than the delivered
``adj_corr`` / ``adj_vol2pk`` cut (their ``last_tr_row`` = last TRAIN target, so they exclude the ~horizon
purge-gap rows between the last train target and the first val target); the difference is a few rows out of
thousands and does not touch val/test data.

Support note (market-basket): ``support(i)``, ``support(j)`` and ``support(i, j)`` for a pair are all taken
over that pair's CO-OBSERVED transaction set (days both stocks traded), not a global day count -- the fair
transaction universe for HNX's heterogeneous listing dates (see ``pairwise_lift``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Thresholds from the paper (single source of truth, recorded in the result's edge_config).
CORR_THRESH = 0.7     # |Pearson rho| edge threshold (paper Eq. 3)
LIFT_THRESH = 1.7     # association-rule lift edge threshold (paper)
# Minimum co-observed days for a stable pair statistic. Matched to the delivered edges so the corr+lift edge
# uses the same overlap discipline: MR.EDGE_MIN_OVERLAP=100 (symmetric corr), MR._MIN_PAIRS=30 (pairwise).
MIN_OVERLAP = 100     # Pearson rho: min co-finite days
MIN_PAIRS = 30        # lift: min co-observed transactions (days both stocks traded)


def load_close_wide(tickers: list[str], price_dir: str | Path) -> pd.DataFrame:
    """Union-date CLOSE panel aligned to ``tickers`` (column order preserved).

    Each column holds a ticker's close on its OWN trading dates and NaN off them (union index). A ticker
    with no ``<ticker>_ohlcv.csv`` becomes an all-NaN column -> a singleton node (self-loop only, no edges).
    """
    price_dir = Path(price_dir)
    series: dict[str, pd.Series] = {}
    for tk in tickers:
        path = price_dir / f"{tk}_ohlcv.csv"
        if not path.exists():
            series[tk] = pd.Series(dtype=float)     # no price file -> singleton
            continue
        raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date", kind="stable")
        raw = raw.drop_duplicates("date", keep="last")
        s = pd.to_numeric(raw["close"], errors="coerce")
        s.index = raw["date"].to_numpy()
        series[tk] = s
    wide = pd.DataFrame(series).reindex(columns=list(tickers)).sort_index()
    return wide


def daily_returns(close_wide: pd.DataFrame) -> np.ndarray:
    """``[T, N]`` simple returns ``r_t = (P_t - P_{t-1})/P_{t-1}`` (paper Eq. 2), computed per ticker on its
    OWN trading dates then placed back on the union index (NaN off own dates / at the first own day).

    Per-ticker own-date differencing avoids spurious returns spanning union dates the ticker did not trade.
    """
    dates = close_wide.index
    out = np.full((len(dates), close_wide.shape[1]), np.nan, dtype=float)
    for j, col in enumerate(close_wide.columns):
        own = close_wide[col].dropna()
        if len(own) < 2:
            continue
        r = own.pct_change().to_numpy(dtype=float)          # first entry NaN
        out[dates.get_indexer(own.index), j] = r
    return out


def pearson_corr(returns: np.ndarray, min_overlap: int = MIN_OVERLAP) -> np.ndarray:
    """Symmetric ``[N, N]`` Pearson rho on co-finite days (paper Eq. 3). NaN off-diagonal when a pair shares
    fewer than ``min_overlap`` co-finite days or either series is constant on the overlap. Diagonal = NaN
    (irrelevant; the caller sets the self-loop)."""
    n = returns.shape[1]
    corr = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        ri = returns[:, i]
        for j in range(i + 1, n):
            rj = returns[:, j]
            m = np.isfinite(ri) & np.isfinite(rj)
            if int(m.sum()) < min_overlap:
                continue
            a, b = ri[m], rj[m]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            rho = float(np.corrcoef(a, b)[0, 1])
            corr[i, j] = corr[j, i] = rho
    return corr


def move_events(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-stock notable-move item + validity mask.

    Threshold for stock ``j`` = the TRAIN median of ``|return_j|`` over its finite days. ``event[t, j]`` is
    True iff ``|return[t, j]| > threshold_j`` (strictly above the stock's typical move). ``valid[t, j]`` is
    True where the return is finite (i.e. the stock traded that day == a transaction). Returns ``(event,
    valid)`` both ``[T, N]`` boolean.
    """
    absr = np.abs(returns)
    valid = np.isfinite(returns)
    n = returns.shape[1]
    event = np.zeros_like(returns, dtype=bool)
    for j in range(n):
        vals = absr[valid[:, j], j]
        if vals.size == 0:
            continue
        thr = float(np.median(vals))
        event[:, j] = valid[:, j] & (absr[:, j] > thr)
    return event, valid


def pairwise_lift(event: np.ndarray, valid: np.ndarray, min_pairs: int = MIN_PAIRS) -> np.ndarray:
    """Symmetric ``[N, N]`` association-rule lift on co-observed days.

    For a pair (i, j) restricted to days BOTH traded (``valid_i & valid_j`` == the shared transaction set):
    ``support_i = mean(event_i)``, ``support_j = mean(event_j)``, ``support_ij = mean(event_i & event_j)``,
    ``lift = support_ij / (support_i * support_j)``. NaN when the pair shares fewer than ``min_pairs``
    co-observed days or either marginal support is 0 (lift undefined). Diagonal = NaN.
    """
    n = event.shape[1]
    lift = np.full((n, n), np.nan, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            co = valid[:, i] & valid[:, j]
            k = int(co.sum())
            if k < min_pairs:
                continue
            ei, ej = event[co, i], event[co, j]
            si, sj = ei.mean(), ej.mean()
            if si == 0.0 or sj == 0.0:
                continue
            sij = float((ei & ej).mean())
            lift[i, j] = lift[j, i] = sij / (si * sj)
    return lift


def build_corrlift_adjacency(
    close_wide: pd.DataFrame,
    cutoff_date,
    corr_thresh: float = CORR_THRESH,
    lift_thresh: float = LIFT_THRESH,
    min_overlap: int = MIN_OVERLAP,
    min_pairs: int = MIN_PAIRS,
) -> tuple[np.ndarray, dict]:
    """Combined corr+lift ``[N, N]`` float32 adjacency (self-loop=1) + an edge-density ``diag`` dict.

    TRAIN-ONLY: only close rows with ``date < cutoff_date`` feed the returns/correlations/supports/lifts,
    then the matrix is frozen. An edge (i, j) is present iff ``|rho_ij| > corr_thresh`` OR ``lift_ij >
    lift_thresh``; its weight is the mean of the fired normalised strengths (``|rho|``; ``lift-1`` MAX-scaled
    by the largest fired excess). Symmetric; diagonal forced to 1.0.
    """
    cutoff = pd.Timestamp(cutoff_date)
    train = close_wide.loc[close_wide.index < cutoff]
    n = close_wide.shape[1]
    returns = daily_returns(train)
    corr = pearson_corr(returns, min_overlap)
    event, valid = move_events(returns)
    lift = pairwise_lift(event, valid, min_pairs)

    corr_fires = np.isfinite(corr) & (np.abs(corr) > corr_thresh)
    lift_fires = np.isfinite(lift) & (lift > lift_thresh)
    np.fill_diagonal(corr_fires, False)
    np.fill_diagonal(lift_fires, False)

    corr_str = np.where(corr_fires, np.abs(corr), 0.0)                 # [0,1]
    lift_excess = np.where(lift_fires, np.nan_to_num(lift) - 1.0, 0.0)  # >0 where fired
    max_excess = lift_excess[lift_fires].max() if lift_fires.any() else 1.0
    lift_str = lift_excess / (max_excess if max_excess > 0 else 1.0)   # MAX-scaled to (0,1]

    present = corr_fires | lift_fires
    n_fired = corr_fires.astype(float) + lift_fires.astype(float)      # 1 or 2 where present
    weight = np.zeros((n, n), dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = (corr_str + lift_str) / np.where(n_fired > 0, n_fired, 1.0)
    weight[present] = w[present].astype(np.float32)
    np.fill_diagonal(weight, 1.0)

    # edge-density breakdown on unordered off-diagonal pairs (i < j)
    iu = np.triu_indices(n, k=1)
    n_corr = int(corr_fires[iu].sum())
    n_lift = int(lift_fires[iu].sum())
    n_either = int(present[iu].sum())
    n_both = int((corr_fires & lift_fires)[iu].sum())
    off = weight.copy()
    np.fill_diagonal(off, 0.0)
    deg = (off != 0).sum(axis=1)
    diag = {
        "n_nodes": n,
        "n_pairs": int(iu[0].size),
        "n_corr_edges": n_corr,
        "n_lift_edges": n_lift,
        "n_either_edges": n_either,
        "n_both_edges": n_both,
        "corr_thresh": corr_thresh,
        "lift_thresh": lift_thresh,
        "min_overlap": min_overlap,
        "min_pairs": min_pairs,
        "avg_off_degree": float(deg.mean()) if n else 0.0,
        "max_off_degree": int(deg.max()) if n else 0,
        "n_singletons": int((deg == 0).sum()),
        "n_train_rows": int(len(train)),
    }
    return weight, diag
