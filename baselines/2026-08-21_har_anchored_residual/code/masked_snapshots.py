"""Masked-panel snapshots (union of dates + node/target masks) — the fix for common-date selection bias.

The common-date snapshot design keeps only dates where EVERY node is observed, which (a) collapses the
test window (VN100 ~49 dates, S&P 500 ~34 before) and (b) selects long-history survivors. This builds the
UNION of dates instead: on each date a node is valid only if its lookback window and target are observed
(``node_mask`` / ``target_mask``); the loss and the graph attention ignore invalid nodes. Edges are a
train-only pairwise-complete correlation Top-K graph (glasso needs a complete PSD covariance, unavailable
on a ragged union panel). Per-ticker scalers are fit on that ticker's TRAIN valid rows only. Purge = h.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission" / "soict_lstm_gat"))
import data_utils as du  # noqa: E402  (reuse har_features)
import pipeline_config as pc  # noqa: E402  (single source of truth for tunable constants)

FIRST_VALID = pc.FIRST_VALID


@dataclass
class MaskedData:
    tickers: list
    adj: np.ndarray            # [N,N] train pairwise-corr Top-K, self-loop, signed
    X_tr: np.ndarray; X_va: np.ndarray; X_te: np.ndarray            # [n,N,seq,3] (0 where invalid)
    nmask_tr: np.ndarray; nmask_va: np.ndarray; nmask_te: np.ndarray  # [n,N] valid INPUT node
    tmask_tr: np.ndarray; tmask_va: np.ndarray; tmask_te: np.ndarray  # [n,N] valid TARGET
    y_tr: np.ndarray; y_va: np.ndarray; y_te: np.ndarray           # [n,N] raw target (nan where invalid)
    har_tr: np.ndarray; har_va: np.ndarray; har_te: np.ndarray     # [n,N,3] raw HAR feats at t
    d_va: list; d_te: list
    t_mean: np.ndarray; t_std: np.ndarray                          # [N] per-node target scaler (train)

    @property
    def N(self):
        return len(self.tickers)


def _load_wide(files):
    series = {}
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        tk = Path(f).name.replace("_processed.csv", "")
        series[tk] = df.set_index("date")["parkinson_volatility"]
    wide = pd.DataFrame(series).sort_index()      # union of dates, NaN where a ticker is missing
    return wide


def build_masked(files, lookback, horizon, train_frac=pc.TRAIN_FRAC, val_frac=pc.VAL_FRAC,
                 min_valid=pc.MIN_VALID_NODES, edge_min_overlap=pc.EDGE_MIN_OVERLAP, top_k=pc.EDGE_TOP_K,
                 min_train_rows=pc.MIN_TRAIN_ROWS):
    wide = _load_wide(files)
    tickers = list(wide.columns)
    dates = wide.index
    pk = wide.to_numpy(float)                       # [T,N] with NaN
    T, N = pk.shape
    feats = np.full((T, N, 3), np.nan)
    for j in range(N):
        feats[:, j, :] = du.har_features(pk[:, j])  # rolling over NaN -> NaN (invalid) at gaps/warmup

    anchors = np.arange(FIRST_VALID + lookback - 1, T - horizon)
    win_ok = np.stack([~np.isnan(feats[t - lookback + 1:t + 1]).any(axis=(0, 2)) for t in anchors])  # [A,N]
    tgt_ok = np.stack([~np.isnan(pk[t + horizon]) for t in anchors])                                  # [A,N]
    node_ok = win_ok & tgt_ok                        # valid target rows (need window too)
    keep = node_ok.sum(1) >= min_valid               # anchors with enough valid nodes
    anchors, win_ok, tgt_ok, node_ok = anchors[keep], win_ok[keep], tgt_ok[keep], node_ok[keep]

    n = len(anchors)
    i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
    sl_tr = slice(0, max(i_tr - horizon, 0)); sl_va = slice(i_tr, max(i_va - horizon, i_tr)); sl_te = slice(i_va, n)

    # DROP nodes with too little TRAIN history: their per-node scaler would be degenerate (default
    # t_std=1.0 vs a real ~1e-4 blows the reconstruction up ~1000x). Keep the union of DATES (the masked
    # benefit); only require each scored NODE to have >= min_train_rows valid train targets.
    train_rows = node_ok[sl_tr].sum(0)                # [N]
    keepn = train_rows >= min_train_rows
    if keepn.sum() < 2:
        keepn = train_rows >= 1
    if not keepn.all():
        kept = [tickers[j] for j in range(N) if keepn[j]]
        wide = wide.loc[:, kept]; tickers = kept
        pk = pk[:, keepn]; feats = feats[:, keepn]
        win_ok = win_ok[:, keepn]; tgt_ok = tgt_ok[:, keepn]; node_ok = node_ok[:, keepn]
        N = int(keepn.sum())
        keep_anchor = node_ok.sum(1) >= min_valid    # re-apply min-valid after node drop
        anchors, win_ok, tgt_ok, node_ok = anchors[keep_anchor], win_ok[keep_anchor], tgt_ok[keep_anchor], node_ok[keep_anchor]
        n = len(anchors)
        i_tr, i_va = int(n * train_frac), int(n * (train_frac + val_frac))
        sl_tr = slice(0, max(i_tr - horizon, 0)); sl_va = slice(i_tr, max(i_va - horizon, i_tr)); sl_te = slice(i_va, n)

    # per-node target scaler on TRAIN valid rows
    tr_anchor = anchors[sl_tr]
    y_tr_full = np.stack([pk[t + horizon] for t in tr_anchor])           # [ntr,N] nan where invalid
    tok_tr = node_ok[sl_tr]
    t_mean = np.array([np.nanmean(y_tr_full[tok_tr[:, j], j]) if tok_tr[:, j].any() else 0.0 for j in range(N)])
    t_std = np.array([np.nanstd(y_tr_full[tok_tr[:, j], j]) if tok_tr[:, j].any() else 1.0 for j in range(N)]) + pc.SCALER_EPS
    # per-node feature scaler on TRAIN valid windows (use feats at anchor t as a proxy row)
    f_tr = np.stack([feats[t] for t in tr_anchor])                      # [ntr,N,3]
    f_mean = np.array([np.nanmean(f_tr[tok_tr[:, j], j], 0) if tok_tr[:, j].any() else np.zeros(3) for j in range(N)])
    f_std = np.array([np.nanstd(f_tr[tok_tr[:, j], j], 0) if tok_tr[:, j].any() else np.ones(3) for j in range(N)]) + pc.SCALER_EPS

    # train-only pairwise-complete correlation Top-K signed edge (rows up to last train target date)
    last_tr_row = int(tr_anchor[-1]) + horizon if len(tr_anchor) else int(anchors[i_tr - 1])
    corr = wide.iloc[:last_tr_row + 1].corr(min_periods=edge_min_overlap).to_numpy().copy()  # writable (pandas 3.x CoW)
    np.fill_diagonal(corr, 0.0)
    corr = np.nan_to_num(corr)
    adj = np.zeros((N, N))
    for i in range(N):
        k = np.argsort(-np.abs(corr[i]))[:top_k]
        adj[i, k] = corr[i, k]
    np.fill_diagonal(adj, 1.0)

    def pack(sl):
        aa = anchors[sl]
        X = np.zeros((len(aa), N, lookback, 3), np.float32)
        # node_mask = valid INPUT window; target_mask = valid window AND valid target (a cell is scored /
        # trained only when BOTH hold — a valid target with a zero-filled invalid window must not enter
        # the loss, HAR fit, or metrics).
        nm = win_ok[sl].astype(np.float32); tm = (win_ok[sl] & tgt_ok[sl]).astype(np.float32)
        y = np.stack([pk[t + horizon] for t in aa])
        har = np.stack([feats[t] for t in aa])
        for a_i, t in enumerate(aa):
            w = feats[t - lookback + 1:t + 1]                            # [lookback,N,3]
            w = np.transpose(w, (1, 0, 2))                               # [N,lookback,3]
            wn = (w - f_mean[:, None, :]) / f_std[:, None, :]
            wn = np.nan_to_num(wn)                                        # invalid nodes -> 0 (masked out)
            X[a_i] = wn
        dts = [dates[t + horizon].strftime("%Y-%m-%d") for t in aa]
        return X, nm, tm, np.nan_to_num(y), np.nan_to_num(har), dts

    Xtr, nmtr, tmtr, ytr, htr, _ = pack(sl_tr)
    Xva, nmva, tmva, yva, hva, dva = pack(sl_va)
    Xte, nmte, tmte, yte, hte, dte = pack(sl_te)
    return MaskedData(tickers, adj.astype(np.float32),
                      Xtr, Xva, Xte, nmtr, nmva, nmte, tmtr, tmva, tmte,
                      ytr, yva, yte, htr, hva, hte, dva, dte, t_mean, t_std)
