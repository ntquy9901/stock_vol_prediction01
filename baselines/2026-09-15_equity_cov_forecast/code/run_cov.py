"""Driver: walk-forward GMV evaluation of the covariance ladder for a market, all rebalance horizons.

Reports, per horizon and model, the OUT-OF-SAMPLE annualized GMV portfolio vol (long-short + long-only),
turnover, and the DM p-value vs the Ledoit-Wolf bar (the model the factor/graph rungs must beat). Adds spike
robustness (re-evaluate excluding regime-shock windows). Writes results/cov_forecast/cov_<market>.json.

Run: python run_cov.py [hose|sp500]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_CODE = Path(__file__).resolve().parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))  # pragma: no cover - path bootstrap

import cov_config as C
import panel as P
import evaluate as E
import gbm_diag as GD
from estimators import ESTIMATORS

REPO = Path(__file__).resolve().parents[3]
BAR = "ledoit_wolf"     # the baseline the other estimators must beat
GBM_COMPOSE = ("ledoit_wolf", "factor_rank_k")   # correlation models to compose with the GBM diagonal (DCC)


def _model_metrics(res: dict, h: int) -> dict:
    """Annualized long-short + long-only GMV vol and turnover from a walk_forward result."""
    return {"ann_vol_ls": E.ann_vol(res["rp_ls"]), "ann_vol_lo": E.ann_vol(res["rp_lo"]),
            "turnover": res["turnover"], "n": int(len(res["rp_ls"]))}


def run(market: str, out_dir=None, load_fn=None) -> dict:
    """Run the covariance ladder walk-forward for `market` across REBALANCE_HORIZONS; write one JSON."""
    out_dir = Path(out_dir) if out_dir else (REPO / "results" / "cov_forecast")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cov_{market}.json"
    R = P.build_panel(market, load_fn=load_fn)
    doc = {"market": market, "n_stocks": R.shape[1], "n_days": R.shape[0],
           "date_range": [str(R.index.min().date()), str(R.index.max().date())],
           "window": C.WINDOW, "bar": BAR, "horizons": list(C.REBALANCE_HORIZONS), "by_h": {}}
    for h in C.REBALANCE_HORIZONS:
        t0 = time.time()
        results = {name: E.walk_forward(R, est, h) for name, est in ESTIMATORS.items()}
        # DCC combination (design section 12): compose the two best correlation models with the GBM per-stock
        # volatility diagonal, if the GBM forecast panel has been built for this (market, h).
        gd = GD.load_diag(market, h, out_dir=out_dir if out_dir else None)
        if gd is not None:
            for name in GBM_COMPOSE:
                results[f"{name}+gbm"] = E.walk_forward(R, ESTIMATORS[name], h, gbm_diag=gd)
        metrics = {name: _model_metrics(res, h) for name, res in results.items()}
        bar = results[BAR]
        dm = {name: E.dm_vol(res["rp_ls"], bar["rp_ls"], res["dates"], h)
              for name, res in results.items() if name != BAR}
        gain = {name: (metrics[BAR]["ann_vol_ls"] - metrics[name]["ann_vol_ls"]) / metrics[BAR]["ann_vol_ls"]
                * 100.0 for name in dm}
        # spike robustness on the winner-vs-bar (best non-bar by ann_vol_ls)
        best = min(dm, key=lambda m: metrics[m]["ann_vol_ls"])
        keep = ~E._spike_mask(results[best]["dates"])
        spike = {"best_model": best,
                 "ann_vol_ls_ex_spike": {m: E.ann_vol(results[m]["rp_ls"][keep]) for m in (best, BAR)},
                 "dm_p_ex_spike": E.dm_vol(results[best]["rp_ls"][keep], bar["rp_ls"][keep],
                                           results[best]["dates"][keep], h),
                 "n_ex_spike": int(keep.sum())}
        beats = {name: bool(gain[name] > 0 and dm[name] < C.DM_ALPHA) for name in dm}
        doc["by_h"][f"h{h}"] = {"metrics": metrics, "dm_vs_bar": dm, "gain_pct_vs_bar": gain,
                                "beats_bar": beats, "spike_robustness": spike}
        best_av = min(metrics, key=lambda m: metrics[m]["ann_vol_ls"])
        print(f"h{h} ({time.time()-t0:.0f}s): "
              + "  ".join(f"{m}={metrics[m]['ann_vol_ls']:.4f}" for m in metrics)
              + f" | best={best_av} | any beats {BAR}: {any(beats.values())}", flush=True)
    out_path.write_text(json.dumps(doc, indent=2))
    print(f"saved {out_path}", flush=True)
    return doc


def main():  # pragma: no cover - entry driver: loads real data + full walk-forward
    ap = argparse.ArgumentParser()
    ap.add_argument("market", nargs="?", choices=("hose", "sp500"), default="hose")
    args = ap.parse_args()
    run(args.market)


if __name__ == "__main__":  # pragma: no cover
    main()
