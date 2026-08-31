"""Enriched-data VN100 feature panel (compute-once causal columns) + per-fold train-only packer.

Unlike ``2026-08-30_walkforward_harx_lstm/wf_panel.build_wf_panel`` (which RECOMPUTES the 5 node
features from raw OHLCV via the delivered helpers), this reader takes the 5 node features DIRECTLY
from the pre-computed causal columns of ``data/processed_enriched/vn100/<ticker>.csv``:

    [parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_{VOLUME_ZSCORE_WINDOW}]

The forecast TARGET is ``parkinson_variance`` at t+h (formed at train time, not stored). The Parkinson
column is a VARIANCE (sigma^2). Leakage-safe: every per-node scaler AND the per-fold vol->PK adjacency
are estimated on the fold's TRAIN window only; forecast/val anchors never enter them.

Reuses the delivered ``masked_rich`` (MR) read-only: ``MaskedRichData`` container, ``_directed_vol2pk``
edge construction, ``N_FEAT``, ``FIRST_VALID``, ``EDGE_TOP_K``. No tunable is hardcoded here -- all
windows / thresholds come from the canonical ``pipeline_config``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "submission" / "soict_lstm_gat",
           _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"):
    sys.path.insert(0, str(_p))   # unconditional (matches wf_panel / run_masked_rich bootstrap style)

import masked_rich as MR      # noqa: E402  (MaskedRichData + _directed_vol2pk + constants, read-only)
import pipeline_config as pc  # noqa: E402  (single source of truth for tunable constants)


def _feature_cols():
    """The 5 enriched node-feature column names, volume window sourced from the canonical config."""
    return ["parkinson_variance", "har_weekly", "har_monthly", "market_pk",
            f"volume_zscore_{pc.VOLUME_ZSCORE_WINDOW}"]


@dataclass
class EnrichedPanel:
    tickers: list
    dates: object              # DatetimeIndex (union of all tickers' dates)
    pk: np.ndarray             # [T,N] Parkinson variance (NaN off a ticker's own dates)
    feats: np.ndarray          # [T,N,5] node features read from enriched columns
    anchors: np.ndarray        # [A] date-index of each kept anchor t
    win_ok: np.ndarray         # [A,N] window (input) valid
    tgt_ok: np.ndarray         # [A,N] target pk[t+h] valid
    node_ok: np.ndarray        # [A,N] win_ok & tgt_ok
    target_dates: np.ndarray   # [A] datetime64 forecast-target date = dates[anchor + horizon]

    @property
    def N(self) -> int:
        return len(self.tickers)


def _check_feature_coverage(feats: np.ndarray, own: np.ndarray, tickers: list) -> None:
    """Fail loud (no silent all-zero degradation) if a whole feature is missing for a valid ticker.

    ``own[:,j]`` marks ticker j's own trading dates. har_weekly/har_monthly/market_pk all-NaN on a
    ticker's own dates means the enriched file is structurally broken for a node that is in the frozen
    universe -- raise rather than train on zeros. volume_zscore all-NaN on own dates would be silently
    imputed to a neutral 0.0 shock, so it is rejected here too (bounded allowlist = none allowed).
    """
    for j, tk in enumerate(tickers):
        oj = own[:, j]
        if not oj.any():
            raise ValueError(f"{tk}: enriched file has no trading dates in the panel")
        for fi, name in ((1, "har_weekly"), (2, "har_monthly"), (3, "market_pk"), (4, "volume_zscore")):
            if np.isnan(feats[oj, j, fi]).all():
                raise ValueError(f"{tk}: enriched feature '{name}' is all-NaN on the ticker's own dates "
                                 f"(missing feature for a valid ticker -- refusing to train on zeros)")


def build_enriched_panel(files, lookback: int, horizon: int, keep_tickers) -> EnrichedPanel:
    """Build the unsplit enriched feature panel over exactly ``keep_tickers``.

    Reads the 5 node features directly from the enriched columns; imputes interior volume_zscore NaN to
    0.0 (neutral shock) on a ticker's own dates only; leaves off-date values NaN (masked out).
    """
    keep = set(keep_tickers)
    cols = _feature_cols()
    frames = {}
    for f in files:
        tk = Path(f).stem
        if tk in keep:
            df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").set_index("date")
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"{tk}: enriched file missing columns {missing}")
            frames[tk] = df[cols]
    tickers = [t for t in keep_tickers if t in frames]
    if len(tickers) < 2:
        raise ValueError(f"build_enriched_panel: {len(tickers)} tickers match keep_tickers (<2)")

    dates = pd.DatetimeIndex(sorted(set().union(*(frames[t].index for t in tickers))))
    T, N = len(dates), len(tickers)
    feats = np.full((T, N, MR.N_FEAT), np.nan)
    for j, tk in enumerate(tickers):
        feats[:, j, :] = frames[tk].reindex(dates)[cols].to_numpy(dtype=float)
    pk = feats[:, :, 0].copy()
    own = ~np.isnan(pk)                                    # [T,N] a ticker's own trading dates

    _check_feature_coverage(feats, own, tickers)
    # impute interior volume_zscore NaN -> 0.0 (neutral shock) on OWN dates only; off-date stays NaN
    vz = feats[:, :, 4]
    feats[:, :, 4] = np.where(own & np.isnan(vz), 0.0, vz)

    anchors = np.arange(MR.FIRST_VALID + lookback - 1, T - horizon)
    win_ok = np.stack([~np.isnan(feats[t - lookback + 1:t + 1]).any(axis=(0, 2)) for t in anchors])
    tgt_ok = np.stack([~np.isnan(pk[t + horizon]) for t in anchors])
    node_ok = win_ok & tgt_ok
    keep_a = node_ok.sum(1) >= pc.MIN_VALID_NODES
    anchors, win_ok, tgt_ok, node_ok = anchors[keep_a], win_ok[keep_a], tgt_ok[keep_a], node_ok[keep_a]
    if len(anchors) == 0:
        raise ValueError(f"build_enriched_panel: no anchor has >= {pc.MIN_VALID_NODES} valid nodes "
                         f"(need at least MIN_VALID_NODES tickers covering the same days)")
    target_dates = dates.to_numpy()[anchors + horizon]
    return EnrichedPanel(tickers, dates, pk, feats, anchors, win_ok, tgt_ok, node_ok, target_dates)


def frozen_universe(files, lookback: int, horizon: int, train_frac: float = pc.TRAIN_FRAC,
                    min_train_rows: int = pc.MIN_TRAIN_ROWS) -> list:
    """The node universe frozen ONCE from a fixed ``train_frac`` split screen (mirrors the delivered
    ``build_masked_rich`` node drop): keep tickers with >= ``min_train_rows`` valid TRAIN anchors."""
    all_tk = sorted({Path(f).stem for f in files if "_rejections" not in Path(f).name})
    panel = build_enriched_panel(files, lookback, horizon, all_tk)
    n = len(panel.anchors)
    i_tr = int(n * train_frac)
    train_rows = panel.node_ok[:i_tr].sum(0)
    keep = [panel.tickers[j] for j in range(panel.N) if train_rows[j] >= min_train_rows]
    if len(keep) < 2:
        raise ValueError(f"frozen_universe: only {len(keep)} tickers pass the {min_train_rows}-train-row screen")
    return keep


def _fit_scalers(panel: EnrichedPanel, train: slice, horizon: int):
    """Per-node target + 5-dim feature scalers on TRAIN valid rows only (forecast/val never enter)."""
    tr_anchor = panel.anchors[train]
    tok = panel.node_ok[train]                              # [ntr,N]
    N = panel.N
    y_tr = np.stack([panel.pk[t + horizon] for t in tr_anchor])            # [ntr,N]
    t_mean = np.array([np.nanmean(y_tr[tok[:, j], j]) if tok[:, j].any() else 0.0 for j in range(N)])
    t_std = np.array([np.nanstd(y_tr[tok[:, j], j]) if tok[:, j].any() else 1.0 for j in range(N)]) + pc.SCALER_EPS
    f_tr = np.stack([panel.feats[t] for t in tr_anchor])                   # [ntr,N,5]
    f_mean = np.array([np.nanmean(f_tr[tok[:, j], j], 0) if tok[:, j].any() else np.zeros(MR.N_FEAT)
                       for j in range(N)])
    f_std = np.array([np.nanstd(f_tr[tok[:, j], j], 0) if tok[:, j].any() else np.ones(MR.N_FEAT)
                      for j in range(N)]) + pc.SCALER_EPS
    return t_mean, t_std, f_mean, f_std


def pack_fold(panel: EnrichedPanel, fold, lookback: int, horizon: int):
    """Build a ``MaskedRichData`` for one fold with TRAIN-only scalers AND a TRAIN-only vol->PK edge."""
    t_mean, t_std, f_mean, f_std = _fit_scalers(panel, fold.train, horizon)
    N = panel.N
    tr_anchor = panel.anchors[fold.train]
    last_tr_row = int(tr_anchor[-1]) + horizon               # last TRAIN target row -> edge sees train only
    adj_vol2pk = MR._directed_vol2pk(panel.feats[:, :, 4], np.sqrt(panel.pk), last_tr_row, MR.EDGE_TOP_K)

    def pack(sl: slice):
        aa = panel.anchors[sl]
        X = np.zeros((len(aa), N, lookback, MR.N_FEAT), np.float32)
        nm = panel.win_ok[sl].astype(np.float32)
        tm = (panel.win_ok[sl] & panel.tgt_ok[sl]).astype(np.float32)
        y = np.stack([panel.pk[t + horizon] for t in aa])
        har = np.stack([panel.feats[t, :, :3] for t in aa])
        har5 = np.stack([panel.feats[t] for t in aa])
        for a_i, t in enumerate(aa):
            w = np.transpose(panel.feats[t - lookback + 1:t + 1], (1, 0, 2))   # [N,lookback,5]
            X[a_i] = np.nan_to_num((w - f_mean[:, None, :]) / f_std[:, None, :])
        dts = [str(np.datetime_as_string(panel.target_dates[i], unit="D"))
               for i in range(*sl.indices(len(panel.anchors)))]
        return X, nm, tm, np.nan_to_num(y), np.nan_to_num(har), np.nan_to_num(har5), dts

    Xtr, nmtr, tmtr, ytr, htr, h5tr, _ = pack(fold.train)
    Xva, nmva, tmva, yva, hva, h5va, dva = pack(fold.val)
    Xte, nmte, tmte, yte, hte, h5te, dte = pack(fold.forecast)
    eye = np.eye(N, dtype=np.float32)                        # adj_corr unused by this experiment
    return MR.MaskedRichData(
        tickers=panel.tickers, adj_vol2pk=adj_vol2pk, adj_corr=eye,
        X_tr=Xtr, X_va=Xva, X_te=Xte,
        nmask_tr=nmtr, nmask_va=nmva, nmask_te=nmte,
        tmask_tr=tmtr, tmask_va=tmva, tmask_te=tmte,
        y_tr=ytr, y_va=yva, y_te=yte,
        har_tr=htr, har_va=hva, har_te=hte,
        d_va=dva, d_te=dte, t_mean=t_mean, t_std=t_std,
        har5_tr=h5tr, har5_va=h5va, har5_te=h5te)
