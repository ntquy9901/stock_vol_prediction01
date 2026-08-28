"""Volatility-proxy robustness: run the FULL pipeline (HAR-X / LSTM / LSTM+GAT, 5 seeds) with the target set
to Parkinson vs Yang-Zhang (yz_daily), on the canonical screened universe, writing to a SEPARATE results tree
results/masked_rich_yz/<target>/ (never touches the delivered masked_rich_floor1e2). Resumable. Throwaway driver
(not committed). Usage: python scripts/eda/run_yz_robustness.py [--targets parkinson yz_daily] [--panels ...]."""
import argparse
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "submission" / "soict_lstm_gat",
           REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code",
           REPO / "scripts" / "garch_masked",
           REPO / "scripts" / "eda"):
    sys.path.insert(0, str(_p))

import run_masked_rich as RM               # noqa: E402
import estimator_forecast_ablation as AB   # noqa: E402
from config import Config                  # noqa: E402


def _done(rp):
    """A YZ cell is complete only if it has a per-seed block for the learned models with a finite QLIKE
    mean (R-12: an empty/partial JSON must not be treated as done). Fail-safe: any error -> not done."""
    if not rp.exists():
        return False
    try:
        res = json.loads(rp.read_text())
        ps = res.get("metrics_per_seed")
        if not isinstance(ps, dict):
            return False
        for model in ("LSTM", "LSTM_wGAT_vol2pk"):
            m = ps.get(model)
            if not isinstance(m, dict) or not np.isfinite(float(m.get("qlike"))):
                return False
        return True
    except (ValueError, TypeError, KeyError, OSError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["parkinson", "yz_daily"])
    ap.add_argument("--panels", nargs="+", default=["vn30", "vn100", "hnx", "hose", "sp500"])
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds (e.g. single seed for slow panels)")
    a = ap.parse_args()
    cfg = replace(Config(), batch_size=32)
    if a.seeds:
        cfg = replace(cfg, seeds=tuple(a.seeds))
    for target in a.targets:
        for panel in a.panels:
            rp = REPO / "results" / "masked_rich_yz" / target / f"{panel}_h{a.horizon}" / "result.json"
            if _done(rp):
                print(f"[yz] {target}/{panel} h{a.horizon} done -> skip", flush=True)
                continue
            keep = AB.screened_tickers(panel)
            price = str(AB.VE.PRICE[panel])
            t0 = time.time()
            with tempfile.TemporaryDirectory() as td:
                files = AB._write_estimator_processed(panel, target, td, keep_tickers=keep)
                if len(files) < 2:
                    print(f"[yz] {target}/{panel}: too few tickers -> skip", flush=True)
                    continue
                res = RM.run(panel, files, price, a.horizon, cfg, with_corr=False,
                             output_param="zscore_floor", out_subdir=f"masked_rich_yz/{target}")
            m = res["metrics"]
            ps = res["metrics_per_seed"]
            print(f"[yz] {target}/{panel} h{a.horizon} {time.time()-t0:.0f}s | N={res['num_nodes']} "
                  f"HAR-X QLIKE={m['HAR-X']['qlike']:.4f} LSTM={ps['LSTM']['qlike']:.4f} "
                  f"GAT={ps['LSTM_wGAT_vol2pk']['qlike']:.4f}", flush=True)
    print("YZ ROBUSTNESS DONE", flush=True)


if __name__ == "__main__":
    main()
