"""Build the per-stock GBM volatility-forecast panel used as the DCC diagonal (design section 12).

At each walk-forward fold, the project's gamma-GBM (FM.gbm) forecasts each stock's h-step-ahead Parkinson
variance from own-history (+ earnings) features, trained causally on data before the fold (with embargo).
sqrt(forecast) = sigma_hat_GBM gives the marginal volatilities; the covariance evaluator then composes
`Sigma = D_GBM R D_GBM`, keeping only the correlation R from a covariance estimator. Output is cached to
`results/cov_forecast/gbm_diag_<market>_h<h>.parquet` (columns: date, ticker, sigma).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_EDA = Path(__file__).resolve().parents[3] / "scripts" / "eda"
_HAR = Path(__file__).resolve().parents[2] / "2026-08-21_har_anchored_residual" / "code"
for _p in (str(_EDA), str(_HAR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402

REPO = Path(__file__).resolve().parents[3]


def _forecast_panel(market: str, h: int, load_fn=None, gbm_fn=None, seeds=None) -> pd.DataFrame:
    """Walk-forward causal per-stock GBM variance forecast -> long df [date, ticker, sigma]. Injectable
    load_fn/gbm_fn/seeds keep it unit-testable without the full data."""
    load_fn = load_fn or FM.load
    gbm_fn = gbm_fn or FM.gbm
    seeds = seeds if seeds is not None else FM.SEEDS
    frames, _sect, edates = load_fn(market)
    a = FM.panel(frames, edates, h)
    has_earn = bool(edates)
    cols = FM.OWN + (FM.EARN if has_earn else [])
    embargo = pd.Timedelta(days=int(h * 1.6) + 5)
    out = []
    for k in range(len(S1.FOLDS) - 1):
        ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
        trf = a[(a.date >= pd.Timestamp(S1.TRAIN_START)) & (a.date < ts - embargo)]
        tef = a[(a.date >= ts) & (a.date < tend)]
        if len(tef) == 0 or len(trf) < 1000:
            continue
        pred = np.mean([gbm_fn(trf, tef, cols, s) for s in seeds], axis=0)
        out.append(pd.DataFrame({"date": tef["date"].to_numpy(), "ticker": tef["ticker"].to_numpy(),
                                 "sigma": np.sqrt(np.maximum(pred, 0.0))}))
    if not out:
        raise ValueError(f"no GBM forecast folds produced for {market!r} h{h}")
    return pd.concat(out, ignore_index=True)


def build(market: str, h: int, out_dir=None, **kw) -> Path:
    """Build + cache the GBM diagonal panel for (market, h); return the parquet path."""
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "cov_forecast")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _forecast_panel(market, h, **kw)
    path = out_dir / f"gbm_diag_{market}_h{h}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_diag(market: str, h: int, out_dir=None) -> pd.DataFrame | None:
    """Load the cached GBM diagonal as a wide [date x ticker] sigma matrix, or None if not built yet."""
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "cov_forecast")
    path = out_dir / f"gbm_diag_{market}_h{h}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.pivot_table(index="date", columns="ticker", values="sigma").sort_index()


def main():  # pragma: no cover - entry driver
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("market", choices=("hose", "sp500"))
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    args = ap.parse_args()
    import cov_config as C
    for h in (args.horizons or C.REBALANCE_HORIZONS):
        p = build(args.market, h)
        print(f"built {p}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
