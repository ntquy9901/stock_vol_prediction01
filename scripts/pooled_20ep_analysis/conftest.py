"""Ensure `from aggregate import ...` resolves when pytest runs from the repo root.

Without this shim the test module only imports when pytest is invoked from inside
this directory; from the repo root collection fails with ModuleNotFoundError.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
