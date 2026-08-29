"""Smoke: CPU dry run + a tiny end-to-end train step of the learned-graph harness on real HNX data.

Kept small (few tickers, SMOKE config) so it exercises the real panel build + one training loop without a
GPU. Tagged so the pre-push smoke gate can select it.
"""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("LEARNED_ABLATION_FORCE_CPU", "1")
_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))


@pytest.mark.smoke
def test_dry_run_builds_panel_and_forward_pass():
    import run_learned_ablation as R
    out = R.run_dry("hnx", horizon=1, max_tickers=10, subgraph_size=5, node_dim=8, alpha=3.0)
    assert out["n_nodes"] >= 2
    assert out["forward_shape"][0] >= 1


@pytest.mark.smoke
def test_dry_run_without_max_tickers_cap(monkeypatch):
    """max_tickers=None -> the universe-cap branch is skipped (kept tiny via a monkeypatched universe)."""
    import run_learned_ablation as R
    import estimator_forecast_ablation as EFA
    small = set(sorted(EFA.screened_tickers("hnx"))[:8])
    monkeypatch.setattr(EFA, "screened_tickers", lambda panel: small)
    out = R.run_dry("hnx", horizon=1, max_tickers=None, subgraph_size=4, node_dim=8, alpha=3.0)
    assert out["n_nodes"] >= 2


@pytest.mark.smoke
def test_tiny_training_produces_gate_compatible_result(tmp_path, monkeypatch):
    import run_learned_ablation as R
    from config import Config
    from dataclasses import replace
    cfg = replace(Config(), epochs=1, min_epochs=1, patience=1, seeds=(42,), batch_size=64)
    # restrict the universe so the smoke stays fast
    import estimator_forecast_ablation as EFA
    keep = set(sorted(EFA.screened_tickers("hnx"))[:10])
    monkeypatch.setattr(EFA, "screened_tickers", lambda panel: keep)
    res = R.run_training("hnx", cfg, 1, subgraph_size=5, node_dim=8, alpha=3.0, out_dir=str(tmp_path))
    written = tmp_path / "learned_graph_ablation_hnx_h1.json"
    assert written.exists()
    # gate-required keys present with fit evidence
    for block in ("metrics", "train_metrics", "val_metrics", "fit_diagnostics"):
        assert R.NO_GRAPH in res[block] and R.STAT in res[block]
        assert R.LEARNED in res[block]
    for name in ("learned_vs_no_graph", "learned_vs_stat_vol2pk", "learned_vs_sector"):
        assert name in res["dm_date_clustered"]
    m = res["metrics"][R.LEARNED]
    assert m["qlike"] > 0 and m["n"] > 0
