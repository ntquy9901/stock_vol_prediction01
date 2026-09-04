import glob
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import pooled_panel as pp  # noqa: E402
import run_pooled_ablation as rp  # noqa: E402
from run_volga_walkforward import enriched_glob  # noqa: E402


def _have(market):
    return bool([f for f in glob.glob(enriched_glob(market)) if "_rejections" not in f])


def test_screened_universe_returns_tickers():
    if not _have("vn30"):
        pytest.skip("enriched vn30 absent")  # pragma: no cover
    uni = pp.screened_universe("vn30", 22, 1)
    assert len(uni) >= 2 and all(isinstance(t, str) for t in uni)


def test_build_screens_maps_and_makes_folds():
    if not (_have("vn30") and _have("vn100")):
        pytest.skip("enriched vn30/vn100 absent")  # pragma: no cover
    panel, folds, wf, cfg, all_idx, score_idx = rp._build(1, folds_target=1, epochs=2)
    assert panel.N >= len(score_idx) >= 2
    assert set(score_idx.tolist()).issubset(range(panel.N))     # VN30 ⊆ VN100 panel
    assert len(all_idx) == panel.N and len(folds) >= 1


def test_build_score_set_is_the_actual_vn30_universe():
    """The scored subset must be the real VN30 frozen universe, not a positional stand-in."""
    if not (_have("vn30") and _have("vn100")):
        pytest.skip("enriched vn30/vn100 absent")  # pragma: no cover
    panel, folds, wf, cfg, all_idx, score_idx = rp._build(1, folds_target=1, epochs=2)
    vn30 = set(pp.screened_universe("vn30", 22, 1))
    scored_tickers = {panel.tickers[j] for j in score_idx.tolist()}
    assert scored_tickers, "no VN30 ticker was scored"
    assert scored_tickers.issubset(vn30)                        # every scored ticker is a real VN30 stock
    assert scored_tickers == vn30 & set(panel.tickers)          # scored == VN30 present in the VN100 panel
