"""Does cross-sectional return dispersion (CSV) help own-history volatility forecasting? (QLIKE + DM + placebo)

Niu et al. (2023, J. Forecasting) show cross-sectional variance of returns — especially its negative-side
component CSV- — predicts aggregate realized volatility under HAR. Here we test it as a MARKET-LEVEL scalar
feature (one value per day, same for every stock) added to the own-history champion. This is NOT a cross-firm
graph (already NO-GO 6x): it is a single causal aggregate.

  CSV-_t = (1/N) * sum_i (r_i,t - rbar_t)^2 * 1[r_i,t < rbar_t]   (below-cross-mean dispersion)
  CSV+_t = (1/N) * sum_i (r_i,t - rbar_t)^2 * 1[r_i,t > rbar_t]   (placebo arm: upside dispersion)
  CSV-_shift = CSV-_t shifted +SHIFT trading days                (MANDATORY date-shifted placebo: a real
               signal must NOT survive a fake alignment — per the 2026-09-10 seasonal-artifact lesson)

Baseline = own(+semi_neg [+EARN on SP500]). Compared per horizon with date-clustered DM + spike-robustness.

Run: ``python csv_dispersion_test.py [hose|sp500]``. Output: results/gamma_gbm/csv_dispersion_<market>.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import config  # noqa: E402
import build_panel as BP  # noqa: E402
import full_matrix as FM  # noqa: E402
import augment_leverage as AL  # noqa: E402  (reuse _pooled, _spike_robust)
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
BASELINE = "base"
OWN = config.own_set(FM.OWN)
CSV_SHIFT = 45   # trading-day shift for the placebo (fake alignment; a real signal must not survive it)


def market_csv(frames):
    """Per-day cross-sectional CSV- / CSV+ (below/above-cross-mean squared dispersion), causal market scalars.
    Returns a DataFrame indexed by date with columns csv_neg, csv_pos, csv_neg_shift."""
    R = pd.DataFrame({tk: f.set_index("date")["daily_return"] for tk, f in frames.items()}).sort_index()
    rbar = R.mean(axis=1)
    dev = R.sub(rbar, axis=0)
    n = R.notna().sum(axis=1).clip(lower=1)
    csv_neg = (dev.where(dev < 0) ** 2).sum(axis=1) / n
    csv_pos = (dev.where(dev > 0) ** 2).sum(axis=1) / n
    out = pd.DataFrame({"csv_neg": csv_neg, "csv_pos": csv_pos})
    out["csv_neg_shift"] = out["csv_neg"].shift(CSV_SHIFT)   # placebo: mis-aligned by SHIFT days
    return out


def feature_frames(frames):
    """semi_neg/semi_pos (BP) per ticker + the shared market CSV columns merged on date."""
    csv = market_csv(frames)
    out = {}
    for tk, f in frames.items():
        d = BP.add_features(f).merge(csv, left_on="date", right_index=True, how="left")
        out[tk] = d
    return out


def _sets(has_earn):
    """base = champion (own+semi_neg [+EARN]); variants add CSV- (real), CSV+ / CSV-_shift (placebos)."""
    base = OWN + ["semi_neg"] + (FM.EARN if has_earn else [])
    return {"base": base, "base+csv_neg": base + ["csv_neg"],
            "base+csv_pos": base + ["csv_pos"], "base+csv_shift": base + ["csv_neg_shift"]}


def run(market, load_fn=None):
    """base vs +CSV features; per-horizon QLIKE + DM(vs base) + spike-robustness + fit. The real signal is
    base+csv_neg; it must beat base AND clearly exceed the csv_pos / csv_shift placebos."""
    load_fn = load_fn or FM.load
    min_rows = config.MIN_ROWS.get(market, config.MIN_ROWS["default"])
    frames, _, edates = load_fn(market)
    frames = feature_frames(frames)
    has_earn = bool(edates)
    sets = _sets(has_earn)
    out = {}
    for h in config.HORIZONS:
        a = FM.panel(frames, edates, h)
        res = {m: AL._pooled(a, cols, h, min_rows) for m, cols in sets.items()}
        if any(res[m] is None for m in sets):
            continue
        y, _, dates, last_tr = res[BASELINE]
        err = {m: M.per_obs_qlike(y, res[m][1], floor=FL) for m in sets}
        q = {m: float(np.mean(err[m])) for m in sets}
        comps, spike = {}, {}
        for m in sets:
            if m == BASELINE:
                continue
            p = float(ST.date_clustered_dm(err[m], err[BASELINE], dates, h)["p_value"])
            comps[m] = {"gain_vs_base_pct": (q[BASELINE] - q[m]) / q[BASELINE] * 100.0, "dm_p": p}
            spike[m] = AL._spike_robust(err[m], err[BASELINE], dates, h)
        tr_q = {m: float(np.mean(M.per_obs_qlike(last_tr["y"].to_numpy(float),
                np.mean([FM.gbm(last_tr, last_tr, cols, s) for s in FM.SEEDS], 0), floor=FL)))
                for m, cols in sets.items()}
        out[f"h{h}"] = {"n": int(len(y)), "has_earn": has_earn, "qlike": q, "vs_base": comps,
                        "spike_robustness": spike,
                        "fit_diagnostics": {m: {"verdict": "overfit" if q[m] > tr_q[m] * 1.25 else "ok",
                                                "train_qlike": tr_q[m], "test_qlike": q[m]} for m in sets}}
    return out


def _print(market, out):  # pragma: no cover - console formatting only
    for h, r in out.items():
        print(f"\n{market} {h} (n={r['n']:,}): base QLIKE={r['qlike']['base']:.4f}", flush=True)
        for m, c in r["vs_base"].items():
            tag = "REAL" if m == "base+csv_neg" else "placebo"
            v = "HELPS" if (c["gain_vs_base_pct"] > 0 and c["dm_p"] < 0.05) else \
                ("helps(ns)" if c["gain_vs_base_pct"] > 0 else "hurts")
            print(f"    {m:16s}[{tag:7s}] QLIKE={r['qlike'][m]:.4f} gain={c['gain_vs_base_pct']:+.3f}% "
                  f"DM p={c['dm_p']:.3g} -> {v}", flush=True)


def main():  # pragma: no cover - entry driver
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"csv_dispersion_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
