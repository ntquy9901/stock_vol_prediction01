"""Strictly point-in-time (PIT) earnings schedule via own-firm reporting cadence.

Backs the paper's leakage-safe earnings robustness (Limitations): instead of the archived *realized*
announcement date (which assumes exact-date foreknowledge at the forecast origin), predict each release
date from the firm's OWN past reporting rhythm -- the previous release plus the median of its PRIOR
inter-release gaps. Only past releases enter, so no future information leaks. Rebuilding the earnings
feature from this predicted schedule and re-scoring gives a conservative lower bound on the earnings gain.

`pit_cadence` is pure and unit-tested; `main` reproduces `results/gamma_gbm/earnings_pit_sp500.json`
(actual-vs-PIT earnings QLIKE gain, SP500 walk-forward, per horizon).
"""
from __future__ import annotations

import numpy as np

MIN_HISTORY = 3   # need >=3 prior releases before the cadence median is meaningful


def pit_cadence(edates):
    """Map each ticker's realized release dates to a strictly-causal PIT-predicted schedule.

    ``edates`` maps ticker -> array-like of release dates. For each event from index ``MIN_HISTORY`` on,
    the predicted date is ``previous_release + median(prior inter-release gaps)`` (only gaps strictly
    before that event, so the prediction is knowable at the forecast origin). The first ``MIN_HISTORY``
    dates and any ticker with fewer than ``MIN_HISTORY + 1`` releases are passed through unchanged.
    Output arrays are sorted ``datetime64[ns]`` to match the panel builder's expected dtype.
    """
    out = {}
    for tk, dates in edates.items():
        d = np.sort(np.asarray(dates).astype("datetime64[D]"))
        if len(d) <= MIN_HISTORY:
            out[tk] = d.astype("datetime64[ns]")
            continue
        gaps = np.diff(d).astype(int)
        preds = list(d[:MIN_HISTORY])
        for i in range(MIN_HISTORY, len(d)):
            step = int(np.median(gaps[:i]))                       # only PRIOR gaps -> causal
            preds.append(d[i - 1] + np.timedelta64(step, "D"))
        out[tk] = np.sort(np.array(preds, dtype="datetime64[D]")).astype("datetime64[ns]")
    return out


def main():  # pragma: no cover - data-driven driver (needs full_matrix + enriched panels)
    import json
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    for p in (str(repo / "baselines" / "2026-09-18_gbm_leaf_graph" / "code"), str(repo / "scripts" / "eda"),
              str(repo / "baselines" / "2026-08-21_har_anchored_residual" / "code"),
              str(repo / "baselines" / "2026-09-13_paper_models" / "code")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import full_matrix as FM
    import leaf_graph as LG
    import metrics as M
    import stats as ST
    import vn_gbm_graph_stage1 as S1
    import config as PMC

    own = PMC.own_set(FM.OWN)
    frames, _sect, ed = FM.load("sp500")
    ed_pit = pit_cadence(ed)
    res = {}
    for h in (1, 5, 10, 22):
        res[str(h)] = {}
        for tag, edx in (("actual", ed), ("pit_cadence", ed_pit)):
            import pandas as pd
            a = FM.panel(frames, edx, h)
            emb = pd.Timedelta(days=int(h * 1.6) + 5)
            eo, ee, dts = [], [], []
            for k in range(len(S1.FOLDS) - 1):
                ts, te = pd.Timestamp(S1.FOLDS[k]), pd.Timestamp(S1.FOLDS[k + 1])
                trf = a[(a.date >= S1.TRAIN_START) & (a.date < ts - emb)]
                tef = a[(a.date >= ts) & (a.date < te)]
                if len(tef) == 0 or len(trf) < 30000:
                    continue
                po = LG.predict_xgb(trf, tef, own, FM.SEEDS)
                pe = LG.predict_xgb(trf, tef, own + FM.EARN, FM.SEEDS)
                y = tef["y"].to_numpy(float)
                eo.append(M.per_obs_qlike(y, po, floor=FM.FL))
                ee.append(M.per_obs_qlike(y, pe, floor=FM.FL))
                dts.append(tef["date"].to_numpy())
            eo, ee, dd = np.concatenate(eo), np.concatenate(ee), np.concatenate(dts)
            qo, qe = float(eo.mean()), float(ee.mean())
            try:
                pval = float(ST.date_clustered_dm(ee, eo, dd, h)["p_value"])
            except Exception:
                pval = None
            res[str(h)][tag] = {"xgb_qlike": qo, "xgbE_qlike": qe,
                                "earn_gain_pct": (qo - qe) / qo * 100, "dm_p": pval}
    (repo / "results" / "gamma_gbm" / "earnings_pit_sp500.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":  # pragma: no cover
    main()
