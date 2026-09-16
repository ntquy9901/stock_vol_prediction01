"""Pytest path bootstrap: expose this baseline's `code/` and the reused har-anchored `stats` module so the
modules import the same way they do under the driver (folder names contain '-', not python-importable)."""
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1] / "code"
_STATS = Path(__file__).resolve().parents[2] / "2026-08-21_har_anchored_residual" / "code"
for p in (str(_CODE), str(_STATS)):
    if p not in sys.path:
        sys.path.insert(0, p)
