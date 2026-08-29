"""Static SECTOR adjacency for the LSTM+GAT volatility model.

Motivation (read-only EDA, docs/reports/2026-08-29_gat_signal_eda_and_harm_analysis.md and the graph
handoff): the shipped statistical edges (Top-K correlation + directed volume-shock->volatility) persist
only ~9-30% train->test, so the frozen graph the model sees at test time barely overlaps the one it was
built on. A GICS-sector edge is static metadata: it does not drift out of sample and carries no
train/test leakage, so it is the natural stable alternative to test against the statistical edge.

The matrix is a drop-in replacement for ``MaskedRichData.adj_vol2pk`` / ``.adj_corr`` -- same [N,N]
float32 convention with a self-loop on the diagonal, unit (unnormalized) weights matching the
statistical adjacencies' self-loop=1.0 convention so ``WeightedGATLayer`` consumes it unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Sentinel prefix that makes each unmapped ticker its OWN singleton sector. Two different unmapped
# tickers must NOT be merged into a single "Unknown" bucket (that would invent cross-stock edges the
# metadata never asserted); each becomes an isolated node with only a self-loop.
_OWN = "__own__:"


def _sector_labels(tickers: list[str], sector_map: dict[str, str]) -> list[str]:
    """Sector label per ticker; unmapped tickers get a unique singleton label (own sector)."""
    return [sector_map.get(t) or (_OWN + t) for t in tickers]


def build_sector_adjacency(
    tickers: list[str],
    sector_map: dict[str, str],
    top_k: int | None = None,
) -> np.ndarray:
    """Same-GICS-sector connectivity matrix aligned to ``tickers`` (node order), self-loop=1.

    ``A[i, j] = 1`` iff ``i`` and ``j`` share a sector (and, with ``top_k``, ``j`` is among the first
    ``top_k`` same-sector neighbours of ``i`` by node order); the diagonal is always 1.

    - ``top_k=None`` (default): fully connected within sector -> symmetric (undirected).
    - ``top_k=K``: cap the number of off-diagonal same-sector neighbours per row to ``K`` using a
      STABLE criterion (the given node order). May be asymmetric when a sector has > K+1 members.

    Weights are unit (0.0 / 1.0) float32, matching the statistical adjacencies' unnormalized
    self-loop=1.0 convention so the same ``WeightedGATLayer`` masking/attention applies unchanged.
    Unmapped tickers form singleton own-sectors (self-loop only, no cross edges).
    """
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must be None or >= 0, got {top_k}")
    labels = _sector_labels(tickers, sector_map)
    n = len(tickers)
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a[i, i] = 1.0                      # self-loop always present
        if top_k is None:
            for j in range(n):
                if j != i and labels[j] == labels[i]:
                    a[i, j] = 1.0
        else:
            kept = 0
            for j in range(n):             # node order == the stable Top-K criterion
                if kept >= top_k:
                    break
                if j != i and labels[j] == labels[i]:
                    a[i, j] = 1.0
                    kept += 1
    return a


def load_sector_map(csv_path: str | Path) -> dict[str, str]:
    """Load a ``ticker,sector`` mapping from the provenance CSV written by ``fetch_sectors.py``.

    Extra columns (source_url, fetched_date, security, ...) are ignored. Blank sector cells are
    treated as unmapped (dropped), so the ticker falls back to a singleton own-sector.
    """
    out: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tk = (row.get("ticker") or "").strip()
            sec = (row.get("sector") or "").strip()
            if tk and sec:
                out[tk] = sec
    return out


def coverage(tickers: list[str], sector_map: dict[str, str]) -> dict[str, object]:
    """Sector-label coverage + degree statistics of the fully-connected sector graph for ``tickers``."""
    labels = _sector_labels(tickers, sector_map)
    mapped = [t for t in tickers if t in sector_map]
    real_sectors = sorted({sector_map[t] for t in mapped})
    a = build_sector_adjacency(tickers, sector_map)
    off = a.copy()
    np.fill_diagonal(off, 0.0)
    deg = off.sum(axis=1)
    return {
        "n_tickers": len(tickers),
        "n_mapped": len(mapped),
        "coverage_frac": (len(mapped) / len(tickers)) if tickers else 0.0,
        "n_sectors": len(real_sectors),
        "sectors": real_sectors,
        "avg_off_degree": float(deg.mean()) if len(deg) else 0.0,
        "max_off_degree": float(deg.max()) if len(deg) else 0.0,
        "n_singletons": int((deg == 0).sum()),
        "labels": dict(zip(tickers, labels)),
    }
