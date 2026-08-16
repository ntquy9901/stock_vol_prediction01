import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
_ROOT = Path(__file__).resolve().parents[3]
# features imports data/scaling from the pooled-news baseline; add both code dirs (as the runners do)
for _p in (CODE, _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features import _check_price_coverage  # noqa: E402


def _touch(d: Path, tickers):
    for t in tickers:
        (d / f"{t}_ohlcv.csv").write_text("date,open,high,low,close,volume\n", encoding="utf-8")


def test_passes_when_all_tickers_have_price(tmp_path):
    _touch(tmp_path, ["ACB", "VNM", "FPT"])
    _check_price_coverage(["ACB", "VNM", "FPT"], tmp_path)   # no raise


def test_tolerates_one_missing(tmp_path):
    _touch(tmp_path, ["ACB", "VNM"])
    _check_price_coverage(["ACB", "VNM", "LPB"], tmp_path)   # 1 missing (LPB) -> allowed


def test_raises_on_mass_missing_config_mismatch(tmp_path):
    _touch(tmp_path, ["ACB"])                                 # only VN30-ish present
    tickers = ["ACB"] + [f"MID{i}" for i in range(71)]        # 71 missing -> config error
    with pytest.raises(ValueError, match="no .*_ohlcv.csv"):
        _check_price_coverage(tickers, tmp_path)
