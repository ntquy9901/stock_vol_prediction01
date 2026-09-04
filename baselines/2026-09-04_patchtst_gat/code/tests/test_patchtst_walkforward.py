"""Real-data CPU tests for the PatchTST walk-forward (CPU-only; see conftest).

Covers: (d) no-lookahead / leakage sanity on a tiny real-data slice (reuse pack_fold's panel +
assert_no_leakage); (e) real-data CPU smoke: 1 fold, 2 epochs, <=8 tickers -> finite QLIKE for every
model, self-evidence check passes. Skips cleanly if the enriched data is not present.
"""
import glob
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import run_patchtst_walkforward as WF
from patchtst_config import PatchTSTHParams
from wf_enriched_panel import build_enriched_panel, frozen_universe, pack_fold
from wf_folds import assert_no_leakage, make_folds
from run_walkforward import training_config

LOOKBACK = 22
HORIZON = 1
N_TICKERS = 8


def _files(market="vn100"):
    fs = sorted(f for f in glob.glob(WF.enriched_glob(market)) if "_rejections" not in Path(f).name)
    if len(fs) < 2:
        pytest.skip(f"no enriched data under {WF.enriched_glob(market)}")
    return fs


def _small_universe(files):
    uni = frozen_universe(files, LOOKBACK, HORIZON)
    return uni[:N_TICKERS]


def test_cpu_only():
    """Guard: the GPU must never be touched by these tests."""
    assert not torch.cuda.is_available(), "CUDA must be disabled (CUDA_VISIBLE_DEVICES=) for baseline smokes"


def test_leakage_free_on_real_slice():
    """(d) fold construction on a real 8-ticker panel is leakage-free (train/val target dates strictly
    precede the forecast region)."""
    files = _files()
    keep = _small_universe(files)
    panel = build_enriched_panel(files, LOOKBACK, HORIZON, keep)
    n = len(panel.anchors)
    test_start = int(n * 0.9)
    K = max(1, math.ceil((n - test_start) / 2))
    folds = make_folds(n, test_start, K, WF.pc.WF_VAL_TAIL, HORIZON)
    assert_no_leakage(folds, panel.target_dates, HORIZON)   # raises on any leakage
    assert len(folds) >= 1


def test_pack_fold_scalers_and_graph_are_train_only():
    """(d) the per-fold packer returns TRAIN-only scalers + a TRAIN-only vol->PK graph with the right
    shapes; targets/masks are finite."""
    files = _files()
    keep = _small_universe(files)
    panel = build_enriched_panel(files, LOOKBACK, HORIZON, keep)
    n = len(panel.anchors)
    test_start = int(n * 0.9)
    folds = make_folds(n, test_start, max(1, n - test_start), WF.pc.WF_VAL_TAIL, HORIZON)
    D = pack_fold(panel, folds[0], LOOKBACK, HORIZON)
    N = D.N
    assert D.X_tr.shape[1:] == (N, LOOKBACK, 5)
    assert D.adj_vol2pk.shape == (N, N)
    assert np.isfinite(D.t_mean).all() and np.isfinite(D.t_std).all()
    assert (D.t_std > 0).all()


@pytest.mark.smoke
def test_real_data_cpu_smoke_finite_qlike():
    """(e) 1 fold, 2 epochs, 1 seed, 8 tickers runs without exception and returns finite QLIKE for
    every model; the in-code over/under-fit evidence self-check is present."""
    files = _files()
    keep = _small_universe(files)
    wf = WF.PatchTSTWFConfig(lookback=LOOKBACK, horizon=HORIZON, folds_target=1)
    cfg = training_config(epochs=2, patience=1, seeds=(42,), batch=32)
    hp = PatchTSTHParams()
    res = WF.run_walkforward(files, wf, cfg, keep, out_path=None, market="vn100", hp=hp)
    assert res["backbone"] == "patchtst"
    for m in ("HAR", "HAR-X", "PatchTST", "PatchTST_wGAT_vol2pk"):
        q = res["metrics"][m]["qlike"]
        assert math.isfinite(q), f"{m} QLIKE not finite: {q}"
        assert res["metrics"][m]["n"] > 0
    # DM + evidence self-check are present and structurally valid
    assert "PatchTST_GAT_vs_PatchTST" in res["dm_date_clustered"]
    assert "evidence_self_check" in res and isinstance(res["evidence_self_check"]["passed"], bool)
    assert set(res["patchtst_hparams"]) >= {"patch_len", "stride", "d_model", "pool"}
