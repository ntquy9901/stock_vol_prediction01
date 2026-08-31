"""Put the baseline ``code/`` dir (enrich, cli, report) and this ``tests/`` dir (_synth) on sys.path."""
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_CODE = _TESTS.parent
for _p in (str(_CODE), str(_TESTS)):
    if _p not in sys.path:  # pragma: no cover - test bootstrap
        sys.path.insert(0, _p)
