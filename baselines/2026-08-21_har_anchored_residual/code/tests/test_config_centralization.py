"""Config-centralization freeze + integration coverage for the HAR-anchored code modules.

Two jobs:
  1. VALUE-FREEZE: every module-level constant / build default in the HAR-anchored code re-exports the
     SAME value as the single canonical ``pipeline_config`` (no drift). The one intentional change is
     ``masked_rich._VOL_WIN == 22`` (was 20; delivered paper JSONs used 20).
  2. INTEGRATION COVERAGE: exercise the code paths that read the centralized FLOOR constants inside
     function bodies (``masked_rich`` / ``masked_snapshots`` scaler eps, ``run_masked_rich`` positivity
     floors, ``experts`` pred-floor + scaler eps, ``screen_features`` top_k) on a tiny synthetic panel,
     so the refactor's body edits are covered and proven behaviour-preserving (finite, positive preds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
_REPO = _HERE.parents[4]
_SUB = _REPO / "submission" / "soict_lstm_gat"
for _p in (str(_CODE), str(_SUB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_config as pc  # noqa: E402
import data_utils as du  # noqa: E402
import masked_rich as MR  # noqa: E402
import masked_snapshots as MS  # noqa: E402
import run_masked_rich as RMR  # noqa: E402
import run_experiment as RE  # noqa: E402  (import covers its centralized min_common default)
import experts  # noqa: E402
import screen_features as SF  # noqa: E402
from config import SMOKE  # noqa: E402


# --------------------------- value-freeze (constants source from pipeline_config) ---------------------------

def test_masked_rich_constants_source_from_canonical():
    assert MR._VOL_WIN == pc.VOLUME_ZSCORE_WINDOW == 22          # <-- intentional change 20 -> 22
    assert MR.FIRST_VALID == pc.FIRST_VALID
    assert MR.N_FEAT == pc.N_NODE_FEATURES
    assert MR.EDGE_TOP_K == pc.EDGE_TOP_K
    assert MR.EDGE_MIN_OVERLAP == pc.EDGE_MIN_OVERLAP
    assert MR._MIN_PAIRS == pc.EDGE_MIN_PAIRS_DIRECTED
    assert MR._MIN_VOL_COVERAGE == pc.MIN_VOL_COVERAGE
    assert MR._EMPTY_VOL_COVERAGE == pc.EMPTY_VOL_COVERAGE


def test_screen_and_snapshot_constants_source_from_canonical():
    assert MS.FIRST_VALID == pc.FIRST_VALID
    assert SF.FIRST_VALID == pc.FIRST_VALID
    assert SF.VOL_WIN == pc.VOLUME_ZSCORE_WINDOW      # same knob as masked_rich (unified at 22)
    assert SF.VOV_WIN == pc.VOL_OF_VOL_WINDOW
    assert SF.HORIZONS == list(pc.HORIZONS)
    assert du.MONTHLY_WIN == pc.HAR_MONTHLY_WINDOW == 22


# --------------------------- synthetic panel ---------------------------

def _synth(tmp_path, n_tickers=12, n_days=440, seed=0):
    """Processed Parkinson CSVs + matching raw OHLCV (with volume) + a data/raw/prices mirror for screen()."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    proc = tmp_path / "proc"; raw = tmp_path / "raw"
    mirror = tmp_path / "data" / "raw" / "prices"
    proc.mkdir(); raw.mkdir(parents=True); mirror.mkdir(parents=True)
    files = []
    for k in range(n_tickers):
        v = np.empty(n_days); v[0] = 1e-4 * (k + 1)
        for t in range(1, n_days):
            v[t] = 5e-5 * (k + 1) + 0.85 * v[t - 1] + 1e-5 * abs(rng.standard_normal())
        pd.DataFrame({"date": dates, "parkinson_volatility": v}).to_csv(proc / f"T{k:02d}_processed.csv", index=False)
        files.append(str(proc / f"T{k:02d}_processed.csv"))
        close = 20.0 + np.cumsum(rng.normal(0, 0.2, n_days))
        span = np.sqrt(v) * close
        ohlcv = pd.DataFrame({"date": dates, "open": close, "high": close + span, "low": close - span,
                              "close": close, "volume": rng.integers(1e5, 1e6, n_days)})
        ohlcv.to_csv(raw / f"T{k:02d}_ohlcv.csv", index=False)
        ohlcv.to_csv(mirror / f"T{k:02d}_ohlcv.csv", index=False)
    return sorted(files), str(raw)


def test_build_masked_rich_body_floors_covered(tmp_path):
    """build_masked_rich runs on synthetic data with volume window=22 -> finite scaled features (covers
    the per-node scaler SCALER_EPS body lines and the volume_zscore path at the new 22-day window)."""
    files, price = _synth(tmp_path)
    D = MR.build_masked_rich(files, price, lookback=10, horizon=1)
    assert np.isfinite(D.X_tr).all() and np.isfinite(D.t_std).all()
    assert (D.t_std > 0).all()                                   # scaler eps keeps std strictly positive
    assert D.X_tr.shape[-1] == pc.N_NODE_FEATURES


def test_build_masked_snapshots_body_floors_covered(tmp_path):
    """masked_snapshots.build_masked runs on synthetic data -> finite 3-feat panel (covers its SCALER_EPS
    body lines and the centralized build defaults)."""
    files, _ = _synth(tmp_path)
    D = MS.build_masked(files, lookback=10, horizon=1)
    assert np.isfinite(D.X_tr).all() and (D.t_std > 0).all()
    assert D.X_tr.shape[-1] == 3


def test_run_masked_rich_positivity_floors_covered(tmp_path, monkeypatch):
    """run() + train_masked_rich reach the POS_FLOOR_FRAC/POS_FLOOR_EPS positivity floors and yield finite,
    strictly-positive predictions (behaviour preserved: floored, never near-zero)."""
    files, price = _synth(tmp_path)
    D = MR.build_masked_rich(files, price, lookback=10, horizon=1)
    p_lstm = RMR.train_masked_rich(D, SMOKE, seed=42, use_graph=False, adj=D.adj_vol2pk)
    assert np.isfinite(p_lstm).all() and (p_lstm > 0).all()
    monkeypatch.setattr(RMR, "REPO", tmp_path)                  # redirect result.json into tmp
    res = RMR.run("synth", files, price, 1, SMOKE, with_corr=False, out_subdir="cfg_test")
    for m in ("HAR", "HAR-X", "LSTM"):
        assert np.isfinite(res["metrics"][m]["qlike"])
    assert (tmp_path / "results" / "cfg_test" / "synth_h1" / "result.json").exists()


def test_experts_pred_floor_and_scaler_eps_covered(tmp_path):
    """experts.build_data + train_neural reach the PRED_FLOOR_FRAC pred-floor and SCALER_EPS scaler lines;
    predictions stay finite and non-negative (floored reconstruction)."""
    files, _ = _synth(tmp_path)
    D = experts.build_data(files, lookback=10, horizon=5, cfg=SMOKE)
    assert D["N"] >= pc.MIN_VALID_NODES
    assert np.all(D["pred_floor"] > 0)                          # PRED_FLOOR_FRAC * mean floor is positive
    te, va, vq, nparam, corr = experts.train_neural(D, SMOKE, seed=42, use_lstm=True, use_graph=False,
                                                    framing="additive")
    preds = np.array([p for _, p in te.values()])
    assert np.isfinite(preds).all() and (preds >= 0).all()


def test_screen_features_topk_body_covered(tmp_path):
    """screen() reaches the centralized EDGE_TOP_K vshock-adjacency call and returns per-horizon results."""
    files, _ = _synth(tmp_path)
    res = SF.screen(files, "vn30", tmp_path, min_common=pc.MIN_COMMON_DATES)
    assert res["vol_win"] == pc.VOLUME_ZSCORE_WINDOW == 22
    assert res["num_nodes"] >= 2 and res["horizons"]           # at least one horizon screened


def test_run_experiment_min_common_default_is_canonical():
    """run_experiment.run sources its min_common default from the canonical config (import-covered line)."""
    import inspect
    assert inspect.signature(RE.run).parameters["min_common"].default == pc.MIN_COMMON_DATES
