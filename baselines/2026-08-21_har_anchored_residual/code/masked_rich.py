"""Masked-panel snapshots with the OLD DELIVERABLE's richer setup — the fairest graph test.

Extends ``masked_snapshots.build_masked`` (union-of-dates panel, node/target masks, per-node train
scalers, purge, min-train-rows node drop) with the two ingredients the deliverable used that the
3-feature binary-mask masked baseline dropped:

  1. **5 node features** ``[pk(t), har_weekly, har_monthly, market_pk(t), volume_zscore(t)]``
     (the deliverable's node vector). ``market_pk`` = cross-sectional MEDIAN of ``sqrt(pk)`` over the
     VALID nodes at day t (causal). ``volume_zscore`` = trailing ``pc.VOLUME_ZSCORE_WINDOW``-day z-score
     of ``log1p(volume)`` per ticker (causal), from raw OHLCV joined to the Parkinson dates. The window
     is CANONICAL 22 (monthly convention); the delivered paper results used 20 (historical -- see
     pipeline_config.VOLUME_ZSCORE_WINDOW).
  2. A **directed volume->PK lead-lag edge** (``A[j,i] = corr(vshock_i(t), sqrt(PK)_j(t+1))`` Top-K
     sources per target, TRAIN dates only, frozen — replicating deliverable ``edges.py``), returned
     ALONGSIDE the symmetric correlation Top-K edge (the existing comparator).

The HAR baseline still uses only the 3 HAR feature columns (daily/weekly/monthly), so it is identical
to the 3-feature masked baseline and directly comparable. Only the LSTM / GAT branches see all 5
features. Leakage-safe: every scaler and both edges are estimated on TRAIN rows only.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import data_utils as du  # noqa: E402  (reuse har_features)
import pipeline_config as pc  # noqa: E402  (single source of truth for tunable constants)

FIRST_VALID = pc.FIRST_VALID
N_FEAT = pc.N_NODE_FEATURES
# volume_zscore trailing window. CANONICAL = 22 (monthly convention, matches har_monthly); the delivered
# paper result JSONs used 20 (see pipeline_config.VOLUME_ZSCORE_WINDOW note -- reproduce with 20).
_VOL_WIN = pc.VOLUME_ZSCORE_WINDOW
# Edge-construction thresholds (single source of truth = pipeline_config, recorded in each result's
# edge_config -- external review M-05). The two edges intentionally use DIFFERENT minimum-overlap
# thresholds: the SYMMETRIC correlation edge needs many overlapping days for a stable Pearson r
# (EDGE_MIN_OVERLAP), while the DIRECTED lead-lag volume->PK edge is a shifted pairwise correlation with
# fewer usable pairs, so it uses a smaller minimum (_MIN_PAIRS).
_MIN_PAIRS = pc.EDGE_MIN_PAIRS_DIRECTED   # directed volume->PK edge: min overlapping (t, t+1) pairs
EDGE_MIN_OVERLAP = pc.EDGE_MIN_OVERLAP    # symmetric correlation edge: min overlapping days for corr()
EDGE_TOP_K = pc.EDGE_TOP_K                # Top-K neighbours per node (both edges)
_MIN_VOL_COVERAGE = pc.MIN_VOL_COVERAGE   # warn if a present ohlcv file covers < this fraction of a ticker's dates
_EMPTY_VOL_COVERAGE = pc.EMPTY_VOL_COVERAGE  # <= this fraction == present-but-empty == semantically missing


@dataclass
class MaskedRichData:
    tickers: list
    adj_vol2pk: np.ndarray     # [N,N] directed train vol->PK Top-K, signed, self-loop
    adj_corr: np.ndarray       # [N,N] symmetric train pairwise-corr Top-K, signed, self-loop
    X_tr: np.ndarray; X_va: np.ndarray; X_te: np.ndarray            # [n,N,seq,5] (0 where invalid)
    nmask_tr: np.ndarray; nmask_va: np.ndarray; nmask_te: np.ndarray  # [n,N] valid INPUT node
    tmask_tr: np.ndarray; tmask_va: np.ndarray; tmask_te: np.ndarray  # [n,N] valid TARGET
    y_tr: np.ndarray; y_va: np.ndarray; y_te: np.ndarray           # [n,N] raw target (0 where invalid)
    har_tr: np.ndarray; har_va: np.ndarray; har_te: np.ndarray     # [n,N,3] raw HAR feats at t (for HAR OLS)
    d_va: list; d_te: list
    t_mean: np.ndarray; t_std: np.ndarray                          # [N] per-node target scaler (train)
    har5_tr: np.ndarray = None; har5_va: np.ndarray = None; har5_te: np.ndarray = None  # [n,N,5] raw 5-feat at t (for the HAR-X fair linear baseline)

    @property
    def N(self):
        return len(self.tickers)


def _load_wide(files):
    """Wide (date x ticker) Parkinson-variance panel; union of dates, NaN where a ticker is missing."""
    series = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        tk = Path(f).name.replace("_processed.csv", "")
        series[tk] = df.set_index("date")["parkinson_volatility"]
    return pd.DataFrame(series).sort_index()


def _volume_zscore_wide(wide: pd.DataFrame, price_dir: Path | str, window: int = _VOL_WIN) -> np.ndarray:
    """[T,N] trailing rolling-``window`` z-score of ``log1p(volume)`` per ticker, on that ticker's OWN
    trading dates then placed back onto the union index (NaN off its own dates, matching the pk panel).

    Causal (trailing window only). A missing ``*_ohlcv.csv`` or a flat/zero-volume window yields 0.0
    (neutral shock) on that ticker's valid rows — it never invents volume and keeps rows eligible.
    """
    dates = wide.index
    out = np.full((len(dates), wide.shape[1]), np.nan, dtype=float)
    missing = []
    low_cover = []
    effectively_absent = []
    for j, tk in enumerate(wide.columns):
        own = dates[wide[tk].notna().to_numpy()]
        if len(own) == 0:
            continue
        path = Path(price_dir) / f"{tk}_ohlcv.csv"
        if path.exists():
            raw = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            vol = raw.set_index("date")["volume"]
            vol = vol[~vol.index.duplicated(keep="last")].reindex(own)
            cover = float(vol.notna().mean())        # M-04: fraction of the ticker's own dates with real volume
            if cover <= _EMPTY_VOL_COVERAGE:
                effectively_absent.append(tk)        # present file but ~no intersecting dates == missing
            elif cover < _MIN_VOL_COVERAGE:
                low_cover.append((tk, round(cover, 3)))
            logv = pd.Series(np.log1p(vol.to_numpy(dtype=float)), index=own)
            mean = logv.rolling(window).mean()
            std = logv.rolling(window).std()
            z = (logv - mean) / std.replace(0.0, np.nan)
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            z = pd.Series(0.0, index=own)   # neutral: no volume series for this ticker
            missing.append(tk)
        out[wide.index.get_indexer(own), j] = z.to_numpy(dtype=float)
    # M-04: surface (do not silently absorb) partial per-date volume coverage -- a present file that covers
    # only part of a ticker's dates turns the missing days into a neutral (zero) shock. Report as a warning
    # rather than fail, since a partial-coverage ticker still has usable volume; the delivered ETL data is
    # ~complete.
    if low_cover:
        warnings.warn(f"volume_zscore: {len(low_cover)} ticker(s) below {_MIN_VOL_COVERAGE:.0%} volume "
                      f"coverage (missing days zero-filled): {low_cover[:5]}", stacklevel=2)
    # M2 + M-04 upper bound (reviewer 2026-08-29): fail loud rather than silently zeroing the volume feature
    # for many tickers (no-silent-degradation). A present file that covers ~no dates is SEMANTICALLY MISSING
    # (all-NaN volume -> all-zero shock), so it counts toward the same bounded allowlist as an absent file --
    # otherwise a present-but-empty file would evade the >2 cap and zero volume for unbounded tickers.
    absent = missing + effectively_absent
    if len(absent) > 2:
        raise ValueError(f"volume_zscore: {len(absent)} tickers have no usable volume under {price_dir} "
                         f"(missing file or present-but-empty; silent-zero volume for >2 not allowed): "
                         f"missing={missing[:5]} empty={effectively_absent[:5]}")
    return out


def _directed_vol2pk(vshock: np.ndarray, sqrt_pk: np.ndarray, last_row: int,
                     top_k: int, min_pairs: int = _MIN_PAIRS) -> np.ndarray:
    """Directed A[target j, source i] = corr(vshock_i(t), sqrt_pk_j(t+1)) over TRAIN dates, Top-K
    |corr| sources per target, self-loop=1 (replicates deliverable edges.build_vol2pk_adjacency)."""
    v = vshock[:last_row + 1]
    p = sqrt_pk[:last_row + 1]
    src = v[:-1]           # source at t
    tgt = p[1:]            # target at t+1
    n = v.shape[1]
    A = np.zeros((n, n), dtype=np.float32)
    for j in range(n):                       # target
        fj = tgt[:, j]
        corrs = np.full(n, np.nan)
        for i in range(n):                   # source
            if i == j:
                continue
            si = src[:, i]
            m = np.isfinite(si) & np.isfinite(fj)
            if int(m.sum()) < min_pairs:
                continue
            a, b = si[m], fj[m]
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            corrs[i] = float(np.corrcoef(a, b)[0, 1])
        valid = np.flatnonzero(np.isfinite(corrs))
        if valid.size:
            k = valid[np.argsort(-np.abs(corrs[valid]))[:top_k]]
            A[j, k] = corrs[k]
    np.fill_diagonal(A, 1.0)
    return A


def build_masked_rich(files, price_dir, lookback, horizon, train_frac=pc.TRAIN_FRAC, val_frac=pc.VAL_FRAC,
                      min_valid=pc.MIN_VALID_NODES, edge_min_overlap=EDGE_MIN_OVERLAP, top_k=EDGE_TOP_K,
                      min_train_rows=pc.MIN_TRAIN_ROWS):
    wide = _load_wide(files)
    tickers = list(wide.columns)
    dates = wide.index
    pk = wide.to_numpy(float)                       # [T,N] with NaN
    T, N = pk.shape

    # --- 5 node features -------------------------------------------------------------------------
    market_pk = np.nanmedian(np.sqrt(pk), axis=1)  # [T] cross-sectional median of sqrt(pk) over valid nodes
    vol_z = _volume_zscore_wide(wide, price_dir)   # [T,N]
    feats = np.full((T, N, N_FEAT), np.nan)
    for j in range(N):
        feats[:, j, :3] = du.har_features(pk[:, j])            # daily/weekly/monthly (rolling -> NaN at gaps)
        feats[:, j, 3] = market_pk                              # shared cross-sectional market factor
        feats[:, j, 4] = vol_z[:, j]                            # per-ticker volume shock (NaN off own dates)

    anchors = np.arange(FIRST_VALID + lookback - 1, T - horizon)
    win_ok = np.stack([~np.isnan(feats[t - lookback + 1:t + 1]).any(axis=(0, 2)) for t in anchors])  # [A,N]
    tgt_ok = np.stack([~np.isnan(pk[t + horizon]) for t in anchors])                                  # [A,N]
    node_ok = win_ok & tgt_ok
    keep = node_ok.sum(1) >= min_valid
    anchors, win_ok, tgt_ok, node_ok = anchors[keep], win_ok[keep], tgt_ok[keep], node_ok[keep]

    n = len(anchors)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))

    # drop nodes with too little TRAIN history (degenerate per-node scaler), keep the union of dates
    def _slices(nn):
        it, iv = int(nn * train_frac), int(nn * (train_frac + val_frac))
        return (slice(0, max(it - horizon, 0)), slice(it, max(iv - horizon, it)), slice(iv, nn), it, iv)

    sl_tr, sl_va, sl_te, i_tr, i_va = _slices(n)
    train_rows = node_ok[sl_tr].sum(0)
    keepn = train_rows >= min_train_rows
    if keepn.sum() < 2:
        keepn = train_rows >= 1
    if not keepn.all():
        kept = [tickers[j] for j in range(N) if keepn[j]]
        wide = wide.loc[:, kept]; tickers = kept
        pk = pk[:, keepn]; feats = feats[:, keepn, :]; vol_z = vol_z[:, keepn]
        win_ok = win_ok[:, keepn]; tgt_ok = tgt_ok[:, keepn]; node_ok = node_ok[:, keepn]
        N = int(keepn.sum())
        keep_anchor = node_ok.sum(1) >= min_valid
        anchors, win_ok, tgt_ok, node_ok = anchors[keep_anchor], win_ok[keep_anchor], tgt_ok[keep_anchor], node_ok[keep_anchor]
        n = len(anchors)
        sl_tr, sl_va, sl_te, i_tr, i_va = _slices(n)

    tr_anchor = anchors[sl_tr]
    # per-node target scaler on TRAIN valid rows
    y_tr_full = np.stack([pk[t + horizon] for t in tr_anchor])
    tok_tr = node_ok[sl_tr]
    t_mean = np.array([np.nanmean(y_tr_full[tok_tr[:, j], j]) if tok_tr[:, j].any() else 0.0 for j in range(N)])
    t_std = np.array([np.nanstd(y_tr_full[tok_tr[:, j], j]) if tok_tr[:, j].any() else 1.0 for j in range(N)]) + pc.SCALER_EPS
    # per-node feature scaler (5-dim) on TRAIN valid anchor rows
    f_tr = np.stack([feats[t] for t in tr_anchor])                      # [ntr,N,5]
    f_mean = np.array([np.nanmean(f_tr[tok_tr[:, j], j], 0) if tok_tr[:, j].any() else np.zeros(N_FEAT) for j in range(N)])
    f_std = np.array([np.nanstd(f_tr[tok_tr[:, j], j], 0) if tok_tr[:, j].any() else np.ones(N_FEAT) for j in range(N)]) + pc.SCALER_EPS

    last_tr_row = int(tr_anchor[-1]) + horizon if len(tr_anchor) else int(anchors[i_tr - 1])

    # symmetric correlation Top-K signed edge (train rows only) — the existing comparator
    corr = wide.iloc[:last_tr_row + 1].corr(min_periods=edge_min_overlap).to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr)
    adj_corr = np.zeros((N, N), np.float32)
    for i in range(N):
        k = np.argsort(-np.abs(corr[i]))[:top_k]
        adj_corr[i, k] = corr[i, k]
    np.fill_diagonal(adj_corr, 1.0)

    # directed volume->PK lead-lag edge (train rows only)
    adj_vol2pk = _directed_vol2pk(vol_z, np.sqrt(pk), last_tr_row, top_k)

    def pack(sl):
        aa = anchors[sl]
        X = np.zeros((len(aa), N, lookback, N_FEAT), np.float32)
        nm = win_ok[sl].astype(np.float32)
        tm = (win_ok[sl] & tgt_ok[sl]).astype(np.float32)
        y = np.stack([pk[t + horizon] for t in aa])
        har = np.stack([feats[t, :, :3] for t in aa])              # raw HAR(3) at anchor t
        for a_i, t in enumerate(aa):
            w = feats[t - lookback + 1:t + 1]                       # [lookback,N,5]
            w = np.transpose(w, (1, 0, 2))                          # [N,lookback,5]
            wn = (w - f_mean[:, None, :]) / f_std[:, None, :]
            X[a_i] = np.nan_to_num(wn)                              # invalid nodes -> 0 (masked out)
        dts = [dates[t + horizon].strftime("%Y-%m-%d") for t in aa]
        return X, nm, tm, np.nan_to_num(y), np.nan_to_num(har), dts

    Xtr, nmtr, tmtr, ytr, htr, _ = pack(sl_tr)
    Xva, nmva, tmva, yva, hva, dva = pack(sl_va)
    Xte, nmte, tmte, yte, hte, dte = pack(sl_te)
    h5 = lambda sl: np.nan_to_num(np.stack([feats[t] for t in anchors[sl]]))  # raw 5-feat at t
    return MaskedRichData(tickers, adj_vol2pk, adj_corr.astype(np.float32),
                          Xtr, Xva, Xte, nmtr, nmva, nmte, tmtr, tmva, tmte,
                          ytr, yva, yte, htr, hva, hte, dva, dte, t_mean, t_std,
                          h5(sl_tr), h5(sl_va), h5(sl_te))
