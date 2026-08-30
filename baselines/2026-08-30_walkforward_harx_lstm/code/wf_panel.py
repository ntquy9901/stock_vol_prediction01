"""Union-of-dates VN100 feature panel (unsplit) + per-fold train-only packer.

``build_wf_panel`` computes the SAME 5 node features as the delivered ``masked_rich.build_masked_rich``
(reusing its helpers read-only) but WITHOUT baking in a fixed fraction split, so an expanding-window
walk-forward can carve arbitrary per-fold boundaries. ``pack_fold`` fits per-node feature+target scalers
on the fold's TRAIN anchors only and packs train/val/forecast into a ``MaskedRichData`` the delivered
``train_masked_rich`` consumes unchanged (no graph -> a dummy identity adjacency).

Parkinson column = VARIANCE (sigma^2). Leakage-safe: forecast/val anchors never enter the scaler fit.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "submission" / "soict_lstm_gat",
           _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"):
    sys.path.insert(0, str(_p))   # unconditional (matches run_masked_rich's bootstrap style)

import data_utils as du       # noqa: E402  (har_features, read-only)
import masked_rich as MR      # noqa: E402  (feature helpers + MaskedRichData, read-only)

MIN_VALID = 8                 # keep an anchor only if >= this many nodes are input+target valid


@dataclass
class WFPanel:
    tickers: list
    dates: object              # DatetimeIndex
    pk: np.ndarray             # [T,N] Parkinson variance (NaN off a ticker's own dates)
    feats: np.ndarray          # [T,N,5] node features (NaN at rolling gaps / off dates)
    anchors: np.ndarray        # [A] date-index of each valid anchor t
    win_ok: np.ndarray         # [A,N] window (input) valid
    tgt_ok: np.ndarray         # [A,N] target pk[t+h] valid
    node_ok: np.ndarray        # [A,N] win_ok & tgt_ok
    target_dates: np.ndarray   # [A] datetime64 forecast-target date = dates[anchor + horizon]

    @property
    def N(self) -> int:
        return len(self.tickers)


def build_wf_panel(files, price_dir, lookback: int, horizon: int, keep_tickers) -> WFPanel:
    """Build the unsplit feature panel over exactly ``keep_tickers`` (the frozen 102-node universe)."""
    keep = set(keep_tickers)
    kfiles = [f for f in files if Path(f).name.replace("_processed.csv", "") in keep]
    if len(kfiles) < 2:
        raise ValueError(f"build_wf_panel: {len(kfiles)} files match keep_tickers (<2)")
    wide = MR._load_wide(kfiles)
    tickers = list(wide.columns)
    dates = wide.index
    pk = wide.to_numpy(float)
    T, N = pk.shape
    market_pk = np.nanmedian(np.sqrt(pk), axis=1)          # [T] causal cross-sectional market factor
    vol_z = MR._volume_zscore_wide(wide, price_dir)         # [T,N] causal volume shock
    feats = np.full((T, N, MR.N_FEAT), np.nan)
    for j in range(N):
        feats[:, j, :3] = du.har_features(pk[:, j])        # daily/weekly/monthly HAR
        feats[:, j, 3] = market_pk
        feats[:, j, 4] = vol_z[:, j]
    anchors = np.arange(MR.FIRST_VALID + lookback - 1, T - horizon)
    win_ok = np.stack([~np.isnan(feats[t - lookback + 1:t + 1]).any(axis=(0, 2)) for t in anchors])
    tgt_ok = np.stack([~np.isnan(pk[t + horizon]) for t in anchors])
    node_ok = win_ok & tgt_ok
    keep_a = node_ok.sum(1) >= MIN_VALID
    anchors, win_ok, tgt_ok, node_ok = anchors[keep_a], win_ok[keep_a], tgt_ok[keep_a], node_ok[keep_a]
    target_dates = dates.to_numpy()[anchors + horizon]
    return WFPanel(tickers, dates, pk, feats, anchors, win_ok, tgt_ok, node_ok, target_dates)


def _fit_scalers(panel: WFPanel, train: slice, horizon: int):
    """Per-node target + 5-dim feature scalers on TRAIN valid rows only (forecast/val never enter)."""
    tr_anchor = panel.anchors[train]
    tok = panel.node_ok[train]                              # [ntr,N]
    N = panel.N
    y_tr = np.stack([panel.pk[t + horizon] for t in tr_anchor])            # [ntr,N]
    t_mean = np.array([np.nanmean(y_tr[tok[:, j], j]) if tok[:, j].any() else 0.0 for j in range(N)])
    t_std = np.array([np.nanstd(y_tr[tok[:, j], j]) if tok[:, j].any() else 1.0 for j in range(N)]) + 1e-8
    f_tr = np.stack([panel.feats[t] for t in tr_anchor])                   # [ntr,N,5]
    f_mean = np.array([np.nanmean(f_tr[tok[:, j], j], 0) if tok[:, j].any() else np.zeros(MR.N_FEAT)
                       for j in range(N)])
    f_std = np.array([np.nanstd(f_tr[tok[:, j], j], 0) if tok[:, j].any() else np.ones(MR.N_FEAT)
                      for j in range(N)]) + 1e-8
    return t_mean, t_std, f_mean, f_std


def pack_fold(panel: WFPanel, fold, lookback: int, horizon: int):
    """Build a ``MaskedRichData`` for one fold with TRAIN-only scalers (dummy identity adjacency)."""
    t_mean, t_std, f_mean, f_std = _fit_scalers(panel, fold.train, horizon)
    N = panel.N

    def pack(sl: slice):
        aa = panel.anchors[sl]
        X = np.zeros((len(aa), N, lookback, MR.N_FEAT), np.float32)
        nm = panel.win_ok[sl].astype(np.float32)
        tm = (panel.win_ok[sl] & panel.tgt_ok[sl]).astype(np.float32)
        y = np.stack([panel.pk[t + horizon] for t in aa]) if len(aa) else np.zeros((0, N))
        har = np.stack([panel.feats[t, :, :3] for t in aa]) if len(aa) else np.zeros((0, N, 3))
        har5 = np.stack([panel.feats[t] for t in aa]) if len(aa) else np.zeros((0, N, MR.N_FEAT))
        for a_i, t in enumerate(aa):
            w = np.transpose(panel.feats[t - lookback + 1:t + 1], (1, 0, 2))   # [N,lookback,5]
            X[a_i] = np.nan_to_num((w - f_mean[:, None, :]) / f_std[:, None, :])
        dts = [str(np.datetime_as_string(panel.target_dates[i], unit="D")) for i in _range(sl, len(panel.anchors))]
        return X, nm, tm, np.nan_to_num(y), np.nan_to_num(har), np.nan_to_num(har5), dts

    Xtr, nmtr, tmtr, ytr, htr, h5tr, _ = pack(fold.train)
    Xva, nmva, tmva, yva, hva, h5va, dva = pack(fold.val)
    Xte, nmte, tmte, yte, hte, h5te, dte = pack(fold.forecast)
    eye = np.eye(N, dtype=np.float32)                       # no graph: dummy adjacency (unused branch)
    return MR.MaskedRichData(
        tickers=panel.tickers, adj_vol2pk=eye, adj_corr=eye,
        X_tr=Xtr, X_va=Xva, X_te=Xte,
        nmask_tr=nmtr, nmask_va=nmva, nmask_te=nmte,
        tmask_tr=tmtr, tmask_va=tmva, tmask_te=tmte,
        y_tr=ytr, y_va=yva, y_te=yte,
        har_tr=htr, har_va=hva, har_te=hte,
        d_va=dva, d_te=dte, t_mean=t_mean, t_std=t_std,
        har5_tr=h5tr, har5_va=h5va, har5_te=h5te)


def _range(sl: slice, n: int) -> range:
    return range(*sl.indices(n))
