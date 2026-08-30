"""Make the HAR-anchored ``code`` modules + the submission modules importable by bare name here.

Mirrors ``submission/soict_lstm_gat/conftest.py``: the code imports its siblings (``masked_rich``,
``experts``, ...) and the submission helpers (``baselines``, ``config``, ``pipeline_config``) by bare
module name. Under pytest's importlib mode the repo root sits on ``sys.path`` first, so the repo-root
``baselines/`` DIRECTORY is cached as a namespace package and shadows the submission's ``baselines.py``
(the one with ``har_fit``). Front the submission + code dirs and drop that stale namespace so the real
module wins.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
_SUB = _HERE.parents[4] / "submission" / "soict_lstm_gat"
for _p in (str(_CODE), str(_SUB)):
    sys.path.insert(0, _p)

_stale = sys.modules.get("baselines")
if _stale is not None and getattr(_stale, "__file__", None) is None:  # repo-root namespace pkg (no __file__)
    del sys.modules["baselines"]
