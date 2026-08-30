"""Make the walk-forward ``code`` + submission + HAR-anchored modules importable by bare name here.

Same rationale as the submission conftest: front the needed dirs and drop the stale repo-root
``baselines/`` namespace package so the submission's ``baselines.py`` (with ``har_fit``) wins under
pytest's importlib import mode.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
_REPO = _HERE.parents[4]
for _p in (str(_CODE), str(_REPO / "submission" / "soict_lstm_gat"),
           str(_REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(_REPO / "scripts" / "quality_gate")):
    sys.path.insert(0, _p)

_stale = sys.modules.get("baselines")
if _stale is not None and getattr(_stale, "__file__", None) is None:
    del sys.modules["baselines"]
