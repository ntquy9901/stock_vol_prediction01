"""Value-freeze: WFConfig field defaults source from the single canonical pipeline_config (no drift)."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
_REPO = _HERE.parents[4]
for _p in (str(_CODE), str(_REPO / "submission" / "soict_lstm_gat"),
           str(_REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(_REPO / "scripts" / "quality_gate")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_config as pc  # noqa: E402
import run_walkforward as WF  # noqa: E402


def test_wfconfig_defaults_source_from_canonical():
    w = WF.WFConfig()
    assert w.lookback == pc.LOOKBACK
    assert w.horizon == pc.WF_HORIZON == 1
    assert w.K == pc.WF_RETRAIN_K == 66
    assert w.val == pc.WF_VAL_TAIL == 66
    assert w.test_frac == pc.WF_TEST_FRAC == 0.90
    assert w.test_start is None
