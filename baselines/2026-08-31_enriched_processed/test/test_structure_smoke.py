"""§3.F structural smoke: the baseline folder is complete and the enrich pipeline boots on a tiny slice."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(BASE / "code" / "tests"))

import enrich  # noqa: E402


@pytest.mark.smoke
def test_baseline_structure_present():
    for sub in ("requirements/requirements.md", "design/design.md", "code/enrich.py",
                "code/cli.py", "code/report.py", "code/tests", "code_review", "test"):
        assert (BASE / sub).exists(), f"missing {sub}"


@pytest.mark.smoke
def test_enrich_boots_on_synthetic_slice():
    from _synth import clean_frame
    out, rej, counts = enrich.build_ticker(clean_frame(n=30, seed=0))
    assert list(out.columns) == enrich.ENRICHED_COLUMNS
    assert len(out) == 30
    pk = out["parkinson_variance"].to_numpy(float)
    assert np.isfinite(pk).all() and (pk >= 0).all()
    assert isinstance(rej, pd.DataFrame)
