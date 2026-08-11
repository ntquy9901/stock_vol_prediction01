"""Path bootstrap so the EDA-GNN baseline tests import both this baseline's code and the pilot."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[3]
_CODE = _HERE.parents[1] / "code"
_PILOT = _HERE.parents[2] / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _p in (str(_CODE), str(_PILOT), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
