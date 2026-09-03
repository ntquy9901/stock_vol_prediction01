import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp  # noqa: E402


def test_vn30_index_maps_and_is_subset():
    class Panel:
        tickers = ["AAA", "FPT", "VIC", "ZZZ"]

    idx = pp.vn30_index(Panel(), ["FPT", "VIC"])
    assert list(idx) == [1, 2]


def test_vn30_index_raises_on_missing():
    class Panel:
        tickers = ["AAA", "FPT"]

    with pytest.raises(ValueError):
        pp.vn30_index(Panel(), ["FPT", "MISSING"])
