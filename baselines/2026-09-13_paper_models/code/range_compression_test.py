"""Does a chart-pattern 'range compression/expansion' (squeeze) feature help beyond own+semi_neg? (QLIKE+DM)

Tests the user's intuition that price patterns (consolidation -> expansion) carry volatility information. We
operationalize the pattern idea as two causal, scale-free range features from daily log-range ln(H/L):
  range_comp = mean(log_range, RANGE_SHORT) / mean(log_range, RANGE_LONG)   (<1 = squeeze / compression)
  range_exp  = log_range_t / mean(log_range, RANGE_LONG)                    (>1 = today already expanding)
and asks whether adding them to the current champion (own + semi_neg [+ earnings on SP500]) lowers OOS QLIKE
under date-clustered DM, with spike-robustness. Prior: near-null (a squeeze ratio is ~ har_weekly/har_monthly,
already available to the tree, and equivalent to a Bollinger-band squeeze already subsumed by GK/RS/YZ).

Run: ``python range_compression_test.py [hose|sp500]``. Output: results/gamma_gbm/range_compression_<market>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

_CODE = Path(__file__).resolve().parent
REPO = _CODE.parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "eda"),
           str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)  # pragma: no cover - path bootstrap (conftest pre-seeds paths under pytest)
import config  # noqa: E402
import build_panel as BP  # noqa: E402
import full_matrix as FM  # noqa: E402
import augment_leverage as AL  # noqa: E402  (reuse _pooled, _spike_robust, _decile_qlike)
import metrics as M  # noqa: E402
import stats as ST  # noqa: E402

FL = FM.FL
BASELINE = "base"
OWN = config.own_set(FM.OWN)


def add_range_features(frame):
    """Add causal range-compression + range-expansion features from log_range (past-only rolling means)."""
    d = frame.sort_values("date").copy()
    lr = d["log_range"]
    long_mean = lr.rolling(config.RANGE_LONG, min_periods=config.SEMI_MIN_PERIODS).mean()
    d["range_comp"] = lr.rolling(config.RANGE_SHORT, min_periods=2).mean() / long_mean
    d["range_exp"] = lr / long_mean
    return d


def feature_frames(frames):
    """semi_neg/semi_pos (BP) + range_comp/range_exp per ticker frame."""
    return {tk: add_range_features(BP.add_features(f)) for tk, f in frames.items()}


def _sets(has_earn):
    """base = champion (own+semi_neg [+EARN]); variants add the range/squeeze features."""
    base = OWN + ["semi_neg"] + (FM.EARN if has_earn else [])
    return {"base": base, "base+comp": base + ["range_comp"], "base+exp": base + ["range_exp"],
            "base+both": base + ["range_comp", "range_exp"]}


def run(market, load_fn=None):
    """base(own+semi_neg) vs +range features; per-horizon QLIKE + DM(vs base) + spike-robustness + fit."""
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
        print(f"\n{market} {h} (n={r['n']:,}): base(own+semi_neg) QLIKE={r['qlike']['base']:.4f}", flush=True)
        for m, c in r["vs_base"].items():
            v = "HELPS" if (c["gain_vs_base_pct"] > 0 and c["dm_p"] < 0.05) else \
                ("helps(ns)" if c["gain_vs_base_pct"] > 0 else "hurts")
            print(f"    {m:12s} QLIKE={r['qlike'][m]:.4f} gain={c['gain_vs_base_pct']:+.3f}% "
                  f"DM p={c['dm_p']:.3g} -> {v}", flush=True)


def main():  # pragma: no cover - entry driver
    market = sys.argv[1] if len(sys.argv) > 1 else "hose"
    out = run(market)
    _print(market, out)
    outp = REPO / "results" / "gamma_gbm" / f"range_compression_{market}.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {outp.relative_to(REPO)}", flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
