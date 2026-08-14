"""Combo-ladder basis + P0 smoke (leakage-safe wiring), per baseline §3.F.

The full P1/P2/P3/G1 training is GPU work validated by the real 3-seed run (results JSON);
these tests cover the NEW glue: the 5-feature + news + directed vol->PK basis and the pure-HAR P0.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import combo_ladder as cl  # noqa: E402

_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


@pytest.fixture(scope="module")
def basis(tmp_path_factory):
    stamp = tmp_path_factory.mktemp("combo") / "stamp"
    return cl.build_basis(stamp)


@pytest.mark.smoke
def test_basis_is_five_feature_news_and_directed_edge(basis):
    pooled, graph, store, allowed, edge_count = basis
    # 5 node features in the nested EDA order, news attached, directed vol->PK edge active
    assert allowed[0].x_price_raw.shape[1] == 5
    assert allowed[0].x_news.shape[1] > 0
    assert edge_count > 0
    # graph-bound train non-empty and val/test observations present
    assert len(allowed) > 0
    assert len(pooled.samples["val"]) > 0 and len(pooled.samples["test"]) > 0


def test_one_basis_invariant_holds(basis):
    # build_basis raises if the graph/pooled observation sets differ; reaching here means it held.
    pooled, graph, store, allowed, edge_count = basis
    for split in ("val", "test"):
        graph_obs = cl._present_obs([s for s in graph.snapshots if s.split == split])
        pooled_obs = {(int(s.key.ticker_id), s.key.target_date) for s in pooled.samples[split]}
        assert graph_obs == pooled_obs


def test_p0_is_pure_har_and_produces_all_metrics(basis, tmp_path):
    pooled, graph, store, allowed, edge_count = basis
    p0 = cl.run_e0(pooled, allowed, store, tmp_path / "P0")
    for split_key in ("validation_metrics", "test_metrics"):
        for metric in _METRICS:
            assert metric in p0[split_key]
            assert np.isfinite(p0[split_key][metric])
