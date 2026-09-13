"""Causal graph-spillover feature block for the hybrid gamma-GBM.

Thin read-only wrapper over the project's shared graph machinery (anti-abstraction gate: reuse, do not
reimplement). The block is the 7-feature ``S1.GRAPH`` set (neighbour-mean volatility, neighbour shock,
neighbour max, neighbour dispersion, node-minus-neighbour, neighbour return, neighbour volume-shock) plus
the single-feature ``g_corr`` (neighbour-mean parkinson_variance) that ``GBM+earn+corr`` uses.

Leakage-safe by construction (proven in ``../test/test_spillover_features.py``):
  * the adjacency ``Wc`` must be built from TRAIN rows only (caller passes ``S1.build_graph(tr, ...)``);
  * every feature at row (date=t, ticker=i) is ``sum_j Wc[i,j] * value_j(t)`` -- a contemporaneous day-t
    cross-section, so perturbing any ``date > t`` row leaves the feature at ``t`` unchanged.
"""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402

SPILL_COLS = list(S1.GRAPH)          # the 7 richer neighbour-aggregation features
CORR_COL = "g_corr"                  # the single-feature spillover (neighbour-mean parkinson_variance)


def add_spillover_features(fold, tickers, Wc):
    """Return a COPY of ``fold`` with the causal spillover block attached.

    ``Wc`` is the row-normalised neighbour weight matrix from ``S1.build_graph`` (TRAIN-only). Adds
    ``CORR_COL`` (via ``FM.nb``) and the 7 ``SPILL_COLS`` (via ``S1.graph_feats``), then fills genuinely
    isolated-ticker / empty-neighbourhood NaNs with 0.0 (bounded, documented -- matches the shared harness).
    Never mutates the caller's frame.
    """
    fold = fold.copy()
    fold[CORR_COL] = FM.nb(fold, tickers, Wc)
    gf = S1.graph_feats(fold, tickers, Wc, "")     # index-aligned DataFrame with columns == SPILL_COLS
    for c in SPILL_COLS:
        fold[c] = gf[c]
    cols = SPILL_COLS + [CORR_COL]
    fold[cols] = fold[cols].fillna(0.0)
    return fold
