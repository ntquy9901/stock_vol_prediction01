"""TDD tests for the sector-adjacency builder (Task 2).

The adjacency is a STATIC same-GICS-sector connectivity matrix aligned to a given ticker order,
with a self-loop on every node. Unlike the statistical edges (Top-K correlation / directed
volume->PK), it is metadata-only: no OOS drift, no train/test leakage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import pytest  # noqa: E402

from sector_adjacency import build_sector_adjacency, coverage, load_sector_map  # noqa: E402


# 4 tickers in 2 sectors -> block-diagonal, self-loops, zero cross-sector.
_TICKERS = ["AAA", "BBB", "CCC", "DDD"]
_SECTORS = {"AAA": "Tech", "BBB": "Tech", "CCC": "Bank", "DDD": "Bank"}


def test_block_diagonal_two_sectors():
    a = build_sector_adjacency(_TICKERS, _SECTORS)
    expected = np.array(
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 1, 1]], dtype=np.float32,
    )
    assert np.array_equal(a, expected)


def test_shape_and_dtype():
    a = build_sector_adjacency(_TICKERS, _SECTORS)
    assert a.shape == (4, 4)
    assert a.dtype == np.float32


def test_self_loop_on_every_node():
    a = build_sector_adjacency(_TICKERS, _SECTORS)
    assert np.array_equal(np.diag(a), np.ones(4, dtype=np.float32))


def test_symmetric_when_undirected():
    a = build_sector_adjacency(_TICKERS, _SECTORS)
    assert np.array_equal(a, a.T)


def test_no_cross_sector_edge():
    a = build_sector_adjacency(_TICKERS, _SECTORS)
    # AAA(Tech) vs CCC(Bank) must be 0 both directions
    assert a[0, 2] == 0.0 and a[2, 0] == 0.0
    assert a[1, 3] == 0.0 and a[3, 1] == 0.0


def test_unmapped_ticker_is_singleton_own_sector():
    # EEE has no sector -> connects only to itself (own singleton sector), never to others.
    tickers = ["AAA", "BBB", "EEE"]
    sectors = {"AAA": "Tech", "BBB": "Tech"}
    a = build_sector_adjacency(tickers, sectors)
    assert a[2, 2] == 1.0
    assert a[2, 0] == 0.0 and a[2, 1] == 0.0
    assert a[0, 2] == 0.0 and a[1, 2] == 0.0
    # the two Tech names still connect
    assert a[0, 1] == 1.0


def test_two_unmapped_tickers_do_not_share_a_sector():
    # Two DIFFERENT unmapped tickers must NOT be grouped into one "Unknown" bucket.
    tickers = ["EEE", "FFF"]
    a = build_sector_adjacency(tickers, {})
    assert a[0, 1] == 0.0 and a[1, 0] == 0.0
    assert a[0, 0] == 1.0 and a[1, 1] == 1.0


def test_topk_caps_degree_and_keeps_self_loop():
    # 5 tickers all same sector; top_k=2 -> each row has self-loop + <=2 same-sector neighbours.
    tickers = [f"T{i}" for i in range(5)]
    sectors = {t: "Tech" for t in tickers}
    a = build_sector_adjacency(tickers, sectors, top_k=2)
    assert a.shape == (5, 5)
    assert np.array_equal(np.diag(a), np.ones(5, dtype=np.float32))
    # off-diagonal degree per row is capped at top_k
    off = a.copy()
    np.fill_diagonal(off, 0.0)
    assert off.sum(axis=1).max() <= 2.0 + 1e-6
    # still no self weight double-counted; values are unit weights
    assert set(np.unique(a).tolist()) <= {0.0, 1.0}


def test_topk_none_is_fully_connected_within_sector():
    tickers = [f"T{i}" for i in range(4)]
    sectors = {t: "Tech" for t in tickers}
    a = build_sector_adjacency(tickers, sectors, top_k=None)
    assert np.array_equal(a, np.ones((4, 4), dtype=np.float32))


def test_topk_zero_is_self_loop_only():
    tickers = [f"T{i}" for i in range(3)]
    sectors = {t: "Tech" for t in tickers}
    a = build_sector_adjacency(tickers, sectors, top_k=0)
    assert np.array_equal(a, np.eye(3, dtype=np.float32))


def test_negative_topk_raises():
    with pytest.raises(ValueError):
        build_sector_adjacency(["A", "B"], {"A": "X", "B": "X"}, top_k=-1)


def test_coverage_reports_mapped_and_degree():
    cov = coverage(_TICKERS, _SECTORS)
    assert cov["n_tickers"] == 4 and cov["n_mapped"] == 4
    assert cov["coverage_frac"] == 1.0
    assert cov["n_sectors"] == 2
    assert cov["avg_off_degree"] == 1.0     # each node has exactly 1 same-sector neighbour
    assert cov["n_singletons"] == 0


def test_coverage_empty_tickers():
    cov = coverage([], {})
    assert cov["n_tickers"] == 0
    assert cov["coverage_frac"] == 0.0
    assert cov["avg_off_degree"] == 0.0
    assert cov["max_off_degree"] == 0.0


def test_coverage_counts_singletons_for_unmapped():
    cov = coverage(["A", "Z"], {"A": "Tech"})   # Z unmapped -> singleton
    assert cov["n_mapped"] == 1
    assert cov["n_singletons"] == 2             # both A (only same-sector member) and Z are isolated


def test_load_sector_map_skips_blank_sector(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("ticker,sector,source_url,fetched_date\n"
                 "AAPL,Information Technology,http://x,2026-08-29\n"
                 "ZZZ,,http://x,2026-08-29\n", encoding="utf-8")
    assert load_sector_map(p) == {"AAPL": "Information Technology"}
