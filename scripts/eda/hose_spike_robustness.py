"""HOSE regime-spike robustness for the frozen paper models (project mandate).

Re-runs the walk-forward GBM(own-8) vs HAR on HOSE saving per-observation (y, pred, date), then checks the
headline QLIKE claim (GBM beats HAR) is not driven by a shock window: (1) per-fold QLIKE, (2) QLIKE + DM after
EXCLUDING the COVID / 2022 / Apr-2025-tariff shock windows (verdict sign must survive), (3) storm-decile QLIKE.

Reuses the exact full_compare / paper machinery read-only. Output: results/gamma_gbm/hose_spike_robustness.json.
Run: ``python hose_spike_robustness.py``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
           str(REPO / "baselines" / "2026-09-13_paper_models" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap
import config as PMCFG  # noqa: E402  (frozen single-source: OWN-8 / horizons / fold gate)
import full_matrix as FM  # noqa: E402
import vn_gbm_graph_stage1 as S1  # noqa: E402
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
OWN = PMCFG.own_set(FM.OWN)                                 # OWN-8 (frozen, rq dropped)
HORIZONS = PMCFG.HORIZONS
MIN_ROWS = PMCFG.MIN_ROWS["default"]
# shock windows (per project rule): COVID crash, 2022 selloff, Apr-2025 tariff shock
SHOCKS = [("2020-02-01", "2020-04-30"), ("2022-01-01", "2022-12-31"), ("2025-04-01", "2025-04-30")]


def _perobs(market="hose"):  # pragma: no cover - real-data walk-forward driver (logic tested via analyze)
    """Per horizon, pooled walk-forward (y, gbm, har, dates, fold_id)."""
    frames, _, _ = FM.load(market)
    out = {}
    for h in HORIZONS:
        a = FM.panel(frames, {}, h)
        embargo = pd.Timedelta(days=int(h * 1.6) + 5)
        yy, gg, hh, dd, ff = [], [], [], [], []
        for k in range(len(S1.FOLDS) - 1):
            ts, tend = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
            tr = a[(a.date >= S1.TRAIN_START) & (a.date < ts - embargo)]
            te = a[(a.date >= ts) & (a.date < tend)]
            if len(te) == 0 or len(tr) < MIN_ROWS:
                continue
            gbm = np.mean([FM.gbm(tr, te, OWN, s) for s in FM.SEEDS], 0)
            har = FM._har_ols(tr, te)
            yy.append(te["y"].to_numpy(float)); gg.append(gbm); hh.append(har)
            dd.append(te["date"].to_numpy()); ff.append(np.full(len(te), k))
        out[h] = (np.concatenate(yy), np.concatenate(gg), np.concatenate(hh),
                  np.concatenate(dd), np.concatenate(ff))
    return out


def _shock_mask(dates):
    """Boolean mask: True where the date falls in ANY shock window."""
    d = pd.to_datetime(dates)
    m = np.zeros(len(d), dtype=bool)
    for lo, hi in SHOCKS:
        m |= (d >= pd.Timestamp(lo)) & (d <= pd.Timestamp(hi))
    return m


def analyze(perobs):
    """Per-fold QLIKE, shock-excluded QLIKE+DM, storm-decile — for each horizon."""
    res = {}
    for h, (y, g, har, dates, fold) in perobs.items():
        eg, eh = M.per_obs_qlike(y, g, floor=FL), M.per_obs_qlike(y, har, floor=FL)
        # (1) per-fold QLIKE
        perfold = {int(k): {"n": int((fold == k).sum()),
                            "gbm": float(eg[fold == k].mean()), "har": float(eh[fold == k].mean())}
                   for k in np.unique(fold)}
        # (2) shock-excluded
        keep = ~_shock_mask(dates)
        dm_all = ST.date_clustered_dm(eg, eh, dates, h)
        dm_ex = ST.date_clustered_dm(eg[keep], eh[keep], dates[keep], h)
        gain_all = (eh.mean() - eg.mean()) / eh.mean() * 100
        gain_ex = (eh[keep].mean() - eg[keep].mean()) / eh[keep].mean() * 100
        # (3) storm decile (by realised y)
        dec = pd.qcut(y, 10, labels=False, duplicates="drop")
        storm = {"top_decile_gbm_qlike": float(eg[dec == dec.max()].mean()),
                 "calm_decile_gbm_qlike": float(eg[dec == 0].mean()),
                 "top_decile_share_of_total": float(eg[dec == dec.max()].sum() / eg.sum())}
        res[f"h{h}"] = {
            "n": int(len(y)), "n_excluded_shock": int((~keep).sum()),
            "qlike_all": {"gbm": float(eg.mean()), "har": float(eh.mean())},
            "qlike_excl_shock": {"gbm": float(eg[keep].mean()), "har": float(eh[keep].mean())},
            "gbm_vs_har_gain_pct": {"all": gain_all, "excl_shock": gain_ex},
            "gbm_vs_har_dm_p": {"all": float(dm_all["p_value"]), "excl_shock": float(dm_ex["p_value"])},
            "verdict_survives": bool(gain_ex > 0 and dm_ex["p_value"] < 0.05),
            "per_fold_qlike": perfold, "storm_decile": storm,
        }
    return res


def main():  # pragma: no cover - entry driver
    res = analyze(_perobs("hose"))
    for h, r in res.items():
        print(f"\n=== HOSE {h} (n={r['n']:,}, shock-excluded {r['n_excluded_shock']:,}) ===", flush=True)
        print(f"  all         : GBM {r['qlike_all']['gbm']:.4f} vs HAR {r['qlike_all']['har']:.4f} "
              f"gain {r['gbm_vs_har_gain_pct']['all']:+.2f}% DM p={r['gbm_vs_har_dm_p']['all']:.2g}", flush=True)
        print(f"  excl shock  : GBM {r['qlike_excl_shock']['gbm']:.4f} vs HAR {r['qlike_excl_shock']['har']:.4f} "
              f"gain {r['gbm_vs_har_gain_pct']['excl_shock']:+.2f}% DM p={r['gbm_vs_har_dm_p']['excl_shock']:.2g} "
              f"-> survives={r['verdict_survives']}", flush=True)
    outp = REPO / "results" / "gamma_gbm" / "hose_spike_robustness.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
