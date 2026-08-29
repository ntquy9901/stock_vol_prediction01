"""Smoke: boot the REAL panel build + a real Graph WaveNet forward pass, and a tiny real ``run_training``
that produces a gate-compatible result.json (few tickers, SMOKE config, CPU) -- no GPU, seconds.
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("GWN_FORCE_CPU", "1")
_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import run_gwn_ablation as R  # noqa: E402
from config import Config  # noqa: E402

PANEL = "hnx"


def _available():
    return R.EFA.screened_tickers(PANEL) is not None


@pytest.mark.smoke
def test_dry_run_builds_panel_and_forwards_both_variants(capsys):
    if not _available():
        pytest.skip(f"{PANEL} panel not available")
    out = R.run_dry(PANEL, horizon=1, max_tickers=8)
    assert out["n_nodes"] >= 2
    printed = capsys.readouterr().out
    assert "GWN_adaptive" in printed and "GWN_no_adaptive" in printed


@pytest.mark.smoke
def test_dry_run_without_max_tickers_cap(monkeypatch, capsys):
    """max_tickers=0 (falsy) skips the cap; a monkeypatched small universe keeps the build tiny while
    exercising the no-cap branch of run_dry."""
    if not _available():
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(R.EFA.screened_tickers(PANEL))[:8])
    monkeypatch.setattr(R.EFA, "screened_tickers", lambda panel: keep)
    out = R.run_dry(PANEL, horizon=1, max_tickers=0)
    assert out["n_nodes"] >= 2
    assert "GWN_adaptive" in capsys.readouterr().out


@pytest.mark.smoke
def test_tiny_training_produces_gate_compatible_result(tmp_path):
    if not _available():
        pytest.skip(f"{PANEL} panel not available")
    keep = set(sorted(R.EFA.screened_tickers(PANEL))[:8])
    monkeyed = R.EFA.screened_tickers
    R.EFA.screened_tickers = lambda panel: keep          # restrict the real build to 8 tickers
    try:
        cfg = replace(Config(), epochs=1, min_epochs=1, patience=1, seeds=(42,))
        res = R.run_training(PANEL, cfg, horizon=1, gwn_batch=16, skip_channels=16, end_channels=32,
                             out_dir=str(tmp_path))
    finally:
        R.EFA.screened_tickers = monkeyed
    assert (tmp_path / "graphwavenet_ablation_hnx_h1.json").exists()
    for k in ("LSTM", "LSTM_wGAT_vol2pk", "GWN_adaptive", "GWN_no_adaptive"):
        assert np.isfinite(res["metrics"][k]["qlike"])
        assert res["fit_diagnostics"][k]["status"] in {"ok", "overfit", "underfit", "unknown"}
    # gate recognizes this as a masked-rich training result and validates the LEARNED keys
    import overfit_check as OF  # noqa: E402
    ok, problems = OF.check_result_evidence(res)
    assert ok, problems
