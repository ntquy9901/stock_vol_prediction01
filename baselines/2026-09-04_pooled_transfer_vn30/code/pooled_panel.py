"""Pooled/transfer ablation helpers for VN30 (single-panel realisation).

ONE VN100 enriched panel + ONE fold set; arms differ only by a training-node mask. Reuses the
delivered VolGA panel reader + masked-rich trainer read-only. No tunable hardcoded here -- windows /
thresholds come from the canonical ``pipeline_config``.
"""
from __future__ import annotations

import glob as _glob
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "baselines" / "2026-08-31_walkforward_volga" / "code",
           _REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           _REPO / "submission" / "soict_lstm_gat"):
    sys.path.insert(0, str(_p))

from run_volga_walkforward import enriched_glob  # noqa: E402
from wf_enriched_panel import frozen_universe  # noqa: E402


def vn30_index(panel, vn30_tickers) -> np.ndarray:
    """Indices of ``vn30_tickers`` within ``panel.tickers`` (raises if any is absent)."""
    pos = {t: j for j, t in enumerate(panel.tickers)}
    missing = [t for t in vn30_tickers if t not in pos]
    if missing:
        raise ValueError(f"VN30 tickers absent from panel: {missing}")
    return np.array([pos[t] for t in vn30_tickers], dtype=int)


def screened_universe(market: str, lookback: int, horizon: int) -> list:
    """``frozen_universe`` over a market's enriched CSVs at the experiment lookback/horizon."""
    files = _glob.glob(enriched_glob(market))
    return frozen_universe(files, lookback, horizon)


def restrict_fold(D, train_idx):
    """Copy of ``MaskedRichData`` ``D`` with train/val masks + vol->PK graph restricted to ``train_idx``.

    Zeros ``tmask_tr/nmask_tr/tmask_va/nmask_va`` columns outside ``train_idx`` (training loss + input
    validity confined to the training universe) and keeps only adjacency edges with BOTH endpoints in
    ``train_idx`` (so restricted-arm nodes attend restricted-arm neighbours only). ``tmask_te`` is left
    unchanged; scoring is handled by ``score_mask``.
    """
    n = D.adj_vol2pk.shape[0]
    keep = np.zeros(n, bool)
    keep[np.asarray(train_idx, int)] = True

    def zc(m):
        m2 = m.copy()
        m2[:, ~keep] = 0.0
        return m2

    adj = D.adj_vol2pk.copy()
    adj[~keep, :] = 0.0
    adj[:, ~keep] = 0.0
    return replace(D, adj_vol2pk=adj,
                   tmask_tr=zc(D.tmask_tr), nmask_tr=zc(D.nmask_tr),
                   tmask_va=zc(D.tmask_va), nmask_va=zc(D.nmask_va))


def score_mask(tmask_te, score_idx):
    """Copy of ``tmask_te`` with all columns outside ``score_idx`` zeroed (score only those nodes)."""
    keep = np.zeros(tmask_te.shape[1], bool)
    keep[np.asarray(score_idx, int)] = True
    m = tmask_te.copy()
    m[:, ~keep] = 0.0
    return m
