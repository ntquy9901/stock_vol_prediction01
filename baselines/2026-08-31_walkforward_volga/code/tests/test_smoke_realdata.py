"""Real-data-sample smoke: build the enriched panel + run a 1-fold, 2-epoch, 1-seed walk-forward on a
small time-slice of the REAL VN100 enriched data (guards encoding / real NaN patterns synthetic data
misses). Also exercises frozen_universe on real files (incl. a rejection-file filter)."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import run_volga_walkforward as WF
import wf_enriched_panel as EP

_REPO = Path(__file__).resolve().parents[4]
_REAL = sorted(f for f in glob.glob(str(_REPO / "data" / "processed_enriched" / "vn100" / "*.csv"))
               if "_rejections" not in Path(f).name)
LB, H = 8, 1


@pytest.fixture
def real_slice(tmp_path):
    """Tail-400-row slice of 10 real enriched tickers written to a tmp dir (+ a dummy rejection file)."""
    if len(_REAL) < 10:  # pragma: no cover - env guard (real enriched data present in this repo)
        pytest.skip("real enriched vn100 data not available")
    files = []
    for f in _REAL[:10]:
        df = pd.read_csv(f, parse_dates=["date"]).tail(400)
        p = tmp_path / Path(f).name
        df.to_csv(p, index=False)
        files.append(str(p))
    (tmp_path / "ZZZ_rejections.csv").write_text("date,foo\n2020-01-01,1\n", encoding="utf-8")
    tickers = [Path(f).stem for f in files]
    return files + [str(tmp_path / "ZZZ_rejections.csv")], tickers


def test_real_slice_reader_and_walkforward(real_slice, tmp_path, train_cfg):
    files, tickers = real_slice
    panel = EP.build_enriched_panel(files, LB, H, tickers)
    assert panel.N == len(tickers) == 10
    own = ~np.isnan(panel.pk)
    assert np.isfinite(panel.feats[:, :, 0][own]).all()               # real parkinson_variance finite
    np.testing.assert_allclose(panel.feats[:, :, 0], panel.pk, equal_nan=True)

    keep = EP.frozen_universe(files, LB, H)                            # rejection file filtered; screen keeps
    assert 2 <= len(keep) <= 10 and "ZZZ_rejections" not in keep

    cfg = train_cfg
    wf = WF.VolgaWFConfig(lookback=LB, horizon=H, folds_target=1, val=30, test_frac=0.85)
    out = tmp_path / "real_res.json"
    res = WF.run_walkforward(files, wf, cfg, tickers, out_path=out)
    assert out.exists() and res["data_source"].endswith("processed_enriched/vn100")
    for m in ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk"):
        assert np.isfinite(res["metrics"][m]["qlike"])
