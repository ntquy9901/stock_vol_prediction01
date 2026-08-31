"""Make the VolGA ``code`` + submission + HAR-anchored + 2026-08-30 walk-forward modules importable
by bare name here, and expose synthetic-data fixtures. Same rationale as the sibling walk-forward
conftest: front the needed dirs and drop the stale repo-root ``baselines`` namespace package so the
submission's ``baselines.py`` (``har_fit``) wins under pytest's importlib import mode.

Shared helpers are exposed as FIXTURES (not module imports) because the repo runs pytest with
``--import-mode=importlib``, under which ``from conftest import ...`` is unreliable.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_CODE = _HERE.parents[1]
_REPO = _HERE.parents[4]
for _p in (str(_CODE), str(_REPO / "submission" / "soict_lstm_gat"),
           str(_REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(_REPO / "baselines" / "2026-08-30_walkforward_harx_lstm" / "code"),
           str(_REPO / "scripts" / "quality_gate")):
    sys.path.insert(0, _p)

_stale = sys.modules.get("baselines")
if _stale is not None and getattr(_stale, "__file__", None) is None:
    del sys.modules["baselines"]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import pipeline_config as pc  # noqa: E402

_VZCOL = f"volume_zscore_{pc.VOLUME_ZSCORE_WINDOW}"


def _write_synth_enriched(dirpath, tickers, T=360, seed=0, drop_col=None):
    """Write tiny synthetic enriched CSVs (the 5 real feature columns + realistic leading-NaN pattern).

    ``drop_col`` omits that column from the LAST ticker's file (to exercise the missing-column guard).
    Returns the list of written file paths.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-02", periods=T)
    pk_all = {tk: np.abs(rng.standard_normal(T)) * 1e-3 + 1e-4 for tk in tickers}
    sqrt_pk = np.stack([np.sqrt(pk_all[tk]) for tk in tickers], axis=1)   # [T,ntk]
    market = np.median(sqrt_pk, axis=1)                                   # cross-sectional (shared per date)
    files = []
    for i, tk in enumerate(tickers):
        pk = pd.Series(pk_all[tk], index=dates)
        vz = pd.Series(rng.standard_normal(T), index=dates)
        vz.iloc[:pc.VOLUME_ZSCORE_WINDOW - 1] = np.nan                    # leading NaN like the real column
        df = pd.DataFrame({
            "date": dates, "parkinson_variance": pk.to_numpy(),
            "har_weekly": pk.rolling(pc.HAR_WEEKLY_WINDOW).mean().to_numpy(),
            "har_monthly": pk.rolling(pc.HAR_MONTHLY_WINDOW).mean().to_numpy(),
            "market_pk": market, _VZCOL: vz.to_numpy()})
        if drop_col is not None and i == len(tickers) - 1:
            df = df.drop(columns=[drop_col])
        p = dirpath / f"{tk}.csv"
        df.to_csv(p, index=False)
        files.append(str(p))
    return files


def _tiny_train_cfg():
    """A fast training config (2 epochs, 1 seed) for fixture-scale runs."""
    from run_volga_walkforward import training_config
    return training_config(epochs=2, patience=1, seeds=(42,), batch=16)


@pytest.fixture
def vzcol():
    return _VZCOL


@pytest.fixture
def synth_writer():
    return _write_synth_enriched


@pytest.fixture
def synth_files(tmp_path):
    """10-ticker synthetic enriched panel (>= MIN_VALID_NODES) written to a tmp dir; (files, tickers)."""
    tickers = [f"T{i:02d}" for i in range(10)]
    return _write_synth_enriched(tmp_path, tickers, T=360, seed=1), tickers


@pytest.fixture
def train_cfg():
    return _tiny_train_cfg()
