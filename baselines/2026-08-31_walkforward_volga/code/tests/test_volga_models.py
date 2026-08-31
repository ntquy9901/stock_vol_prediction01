"""3-model fold + pooled walk-forward on a tiny synthetic panel: HAR-X / LSTM / VolGA all run and
produce finite, floored predictions; run_walkforward pools + writes the evidence schema."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import run_volga_walkforward as WF
import wf_enriched_panel as EP
from wf_folds import make_folds

LB, H = 8, 1
_MODELS = ("HAR", "HAR-X", "LSTM", "LSTM_wGAT_vol2pk")


def test_run_fold_three_models_finite_floored(synth_files, train_cfg):
    files, tickers = synth_files
    panel = EP.build_enriched_panel(files, LB, H, tickers)
    fold = make_folds(len(panel.anchors), int(len(panel.anchors) * 0.7), 40, 25, H)[0]
    cfg = train_cfg
    upd, flats, ev = WF.run_fold(panel, fold, replace(WF.VolgaWFConfig(lookback=LB, horizon=H), val=25), cfg)

    assert set(upd) == {"HAR", "HAR-X", "LSTM_seeds", "VolGA_seeds"}
    assert len(upd["LSTM_seeds"]) == len(upd["VolGA_seeds"]) == len(cfg.seeds)
    # every forecast prediction is finite and strictly positive (floored)
    for name in ("HAR", "HAR-X"):
        preds = np.array([v[1] for v in upd[name].values()])
        assert np.isfinite(preds).all() and (preds > 0).all()
    for seed_dict in upd["VolGA_seeds"]:
        preds = np.array([v[1] for v in seed_dict.values()])
        assert np.isfinite(preds).all() and (preds > 0).all()
    # over/under-fit evidence for all four models
    for block in ("train_metrics", "val_metrics", "test_metrics", "fit_diagnostics", "learning_curves"):
        assert block in ev
    for m in _MODELS:
        assert set(ev["train_metrics"][m]) >= {"qlike", "r2"}
        assert ev["fit_diagnostics"][m]["status"] in ("ok", "overfit", "underfit", "unknown")
    for gm in ("LSTM", "LSTM_wGAT_vol2pk"):
        assert len(ev["learning_curves"][gm]["train"]) == len(cfg.seeds)
    yt, pt = flats["train"]
    assert set(pt) == set(_MODELS) and len(yt) == len(pt["LSTM"])


def test_run_walkforward_pools_and_writes(synth_files, tmp_path, train_cfg):
    files, tickers = synth_files
    cfg = train_cfg
    wf = WF.VolgaWFConfig(lookback=LB, horizon=H, folds_target=2, val=25, test_frac=0.7)
    out = tmp_path / "res.json"
    res = WF.run_walkforward(files, wf, cfg, tickers, out_path=out)     # out_path set + test_start None
    assert out.exists()
    assert res["num_nodes"] == len(tickers) and res["n_folds"] >= 1
    for m in _MODELS:
        assert np.isfinite(res["metrics"][m]["qlike"])
        assert m in res["train_metrics"] and m in res["val_metrics"] and m in res["fit_diagnostics"]
    assert "VolGA_vs_LSTM" in res["dm_date_clustered"]
    assert set(res["learning_curves"]) == {"LSTM", "LSTM_wGAT_vol2pk"}


def test_run_walkforward_explicit_test_start_no_file(synth_files, train_cfg):
    files, tickers = synth_files
    cfg = train_cfg
    wf = replace(WF.VolgaWFConfig(lookback=LB, horizon=H, folds_target=1, val=25), test_start=230)
    res = WF.run_walkforward(files, wf, cfg, tickers, out_path=None)    # out_path None + test_start set
    assert res["test_start_anchor"] == 230 and res["seconds"] >= 0.0
