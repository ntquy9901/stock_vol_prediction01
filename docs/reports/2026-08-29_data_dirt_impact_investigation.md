# Data-Quality "Dirty Data" — Impact Investigation and Decision Justification (2026-08-29)

## Purpose

This document responds to `docs/reports/2026-08-29_raw_processed_data_quality_audit.md` (the raw/processed
data-quality audit) and justifies, with reproducible evidence, the decision **not to re-run** the
volatility-proxy robustness suite (`results/masked_rich_yz/*`) on account of invalid-geometry OHLC bars in the
HNX/HOSE raw data. It is written to be independently verifiable by a reviewing agent: every claim cites the
source line or the exact measurement command that produced the number.

## Verdict

The invalid-geometry OHLC bars in the HNX/HOSE raw data **do not contaminate any reported estimator**. The
estimator pipeline already excludes such bars at the estimator level, using the same OHLC-geometry rule and
relative tolerance as the loader validator. Measured impact: Parkinson (primary target) per-ticker mean shift
median 0.05%–0.07%; Garman–Klass and Rogers–Satchell shift exactly 0.0000%; windowed Yang–Zhang handled
conservatively (windows spanning a corrupt bar are set to NaN). **No re-run is required.** The remaining actions
are disclosure and a rejection manifest, not recomputation.

## 1. Classification of the reported "dirt"

Not all audit findings are data errors that require a fix; they must be separated.

| Audit finding | True nature | Action class |
|---|---|---|
| Raw HNX/HOSE OHLC geometry violations (~0.3% of rows) | Real data error (a small number genuine, e.g. ADC 2018-03-01 `low>open`, PJC 2008-12-08 `high<low`) | Excluded at estimator level (see §3); manifest + optional targeted reprocess |
| HNX ~45.3% Parkinson target `= 0` | **Not an error** — genuine illiquidity (`high == low`, no intraday move) | Disclose (illiquidity), not "fix" |
| S&P 500 strict-OHLC flags (~1e-17) | **Not an error** — adjusted-price floating-point noise (`auto_adjust`) | Handled by relative tolerance (commit `061fc26`) |
| ETL upper clip at `0.1` | Modelling choice (winsorization), affects delivered processed Parkinson only | Record `n_clipped` / sensitivity; disclose |
| VN prices not confirmed split-adjusted | Known limitation (needs adjusted source) | Disclose caveat |

The only class that is a genuine data error is the ~0.3% invalid-geometry rows in HNX/HOSE. The rest is either a
real market feature to disclose or floating-point noise already handled.

## 2. Evidence A — impact of invalid bars on the Parkinson target (primary result)

**Method.** For every HNX/HOSE raw ticker, flag rows violating OHLC geometry with a `1e-6` relative tolerance
(`low>high`, `high<max(open,close)`, `low>min(open,close)`), compute the Parkinson variance
`ln(high/low)^2 / (4 ln 2)`, and compare the per-ticker mean target computed on **all** rows vs on
**geometry-valid** rows only.

**Result.**

| Group | Rows | Invalid (rtol 1e-6) | Per-ticker mean-target shift (drop invalid) | Invalid rows extreme? |
|---|---:|---|---|---|
| HNX | 1,069,449 | 3,441 (0.322%), 189 tickers | median 0.047%, p90 0.820%, max 19.71% | No — median Parkinson `0` (mostly `high==low`) |
| HOSE | 1,388,158 | 3,541 (0.255%), 233 tickers | median 0.073%, p90 1.151%, max 10.70% | No — comparable to valid-row median |

**Interpretation.** Parkinson uses only `high`/`low`. Of the ~3,500 invalid rows per market, the large majority
violate `high<max(open,close)` or `low>min(open,close)` — constraints on **open/close**, which Parkinson does
not use, so they do not distort the Parkinson target at all. Only `high<low` rows (3 per market per the audit
table) distort Parkinson, and those are negligible. Dropping every invalid row shifts the per-ticker mean target
by a median of ~0.05%.

## 3. Evidence B — the estimator pipeline already excludes invalid-geometry bars

The audit's concern for GK/RS/YZ is that they **do** use open/close, so the ~3,500 open/close violations could
distort them. Inspection of the estimator implementation shows they are already masked out.

**Source (`scripts/eda/volatility_estimators.py`, `estimators_from_ohlcv`).**
- Lines 77–80: an `ok` mask enforcing the full OHLC geometry with the same relative tolerance `_OHLC_RTOL`
  (`high>=low`, `high>=max(open,close)*(1-rtol)`, `low<=min(open,close)*(1+rtol)`).
- Line 130: `out.loc[~ok, EST + ["yz_daily","yz_rma20","yang_zhang"]] = np.nan` — every estimator
  (close2close, parkinson, garman_klass, rogers_satchell, rs_overnight, yang_zhang) is set to NaN on any
  invalid-geometry row.
- Lines 111–113: for the windowed Yang–Zhang, invalid rows are masked to NaN **before** the rolling variance,
  so a window spanning a corrupt bar yields NaN rather than a poisoned value (code-review LOW-1).
- Lines 90–92: the overnight return is additionally winsorized to `±0.20` to bound unadjusted-split spikes in
  the overnight-bearing estimators.

**Empirical verification.** Running `estimators_from_ohlcv` on the HNX/HOSE tickers that contain invalid rows,
then reading the estimator values on exactly those invalid rows:

| Group (60 tickers w/ invalid rows) | Finite estimator values on invalid rows | GK shift | RS shift | YZ shift |
|---|---|---|---|---|
| HNX | **0 leaks** (all NaN) | 0.0000% | 0.0000% | median 0.485%, p90 6.34%, max 15.26% |
| HOSE | **0 leaks** (all NaN) | 0.0000% | 0.0000% | median 0.550%, p90 7.78%, max 34.96% |

**Interpretation.**
- **Garman–Klass, Rogers–Satchell (per-day): exactly 0% impact.** Masking a bad row to NaN and skipping it in
  `nanmean` is identical to dropping it; the per-ticker mean is unchanged.
- **Yang–Zhang (windowed, 20 days): the residual "shift" is not an error.** It is the difference between two
  cleaning strategies: (i) the module's — mask the corrupt bar to NaN so any 20-day window containing it is NaN
  (that date is dropped); (ii) an alternative — delete the row first, then compute a "20-day" window over
  non-adjacent days. Strategy (i) is the correct and conservative one: a windowed variance cannot be computed
  cleanly if one day in the window is corrupt, and strategy (ii) silently mixes non-consecutive days. The run
  uses strategy (i).

## 4. Decision

No estimator in the robustness suite is re-run. Every estimator (Parkinson primary + GK/RS/YZ robustness) is
already computed on geometry-valid bars, using the same OHLC rule and tolerance as the loader validator
(hardened in commits `d0ce184` and `061fc26`). The invalid bars do not contaminate any reported number.

## 5. Scope note — delivered processed Parkinson files vs. the estimator run

Two data paths must be distinguished:
- **Estimator robustness run** (`results/masked_rich_yz`, via `estimator_forecast_ablation._write_estimator_processed`
  → `volatility_estimators.estimators_from_ohlcv`): self-masks invalid geometry (§3). Clean.
- **Delivered processed Parkinson files** (`data/processed/...`, via
  `src/common/process_parkinson_pipeline.process_single_stock` → `parkinson_utils`): these were generated before
  the validator was hardened, so they may retain Parkinson values from the ~3 `high<low` rows per HNX/HOSE
  market. The audit verified the 33-ticker top-level set reproduces the formula exactly, and §2 bounds the
  aggregate impact at ~0.05% median. On the next reprocess the hardened `validate_ohlc_data` (now tolerance-aware,
  §Evidence in commits below) rejects those rows and a rejection manifest records them.

## 6. Remaining actions (disclosure / housekeeping only — no recomputation)

1. Emit an ETL rejection manifest (ticker, date, reason) — the `ok` mask already identifies the rows; dump it.
2. Record `n_clipped` / `clip_fraction` for the `0.1` upper clip and the count of YZ dates dropped by NaN windows.
3. Paper disclosures: describe the data as "integrity-checked with market-specific limitations" (not "clean");
   report the HNX zero-range (illiquidity) share with the headline metrics; state that the VN prices are not
   confirmed split-adjusted; state that the target is Parkinson **variance** (σ²), not standard deviation.

## 7. Related fixes already committed

- `d0ce184` — `validate_ohlc_data` rejects impossible OHLC geometry (review HIGH-03) + 9 property tests.
- `061fc26` — the same geometry check made tolerance-aware (`_OHLC_GEOMETRY_RTOL = 1e-6`) so adjusted-price
  floating-point noise is not false-rejected (data audit MEDIUM-DATA-02).
- The validator's rule and tolerance match `volatility_estimators.py`'s `ok` mask, so the loader boundary and
  the estimator boundary now apply the same geometry criterion.

## 8. Reproduction

Both measurements are read-only and do not touch the running suite. `PY` denotes `.venv_gpu_encode/Scripts/python.exe`.

**Evidence A (Parkinson target impact):**
```python
import glob, numpy as np, pandas as pd
LN2x4 = 4*np.log(2); RTOL = 1e-6
for group, ddir in (("HNX","data/raw/prices/hnx_vnstock"),("HOSE","data/raw/prices/hose_vnstock")):
    tot=inval=0; shifts=[]
    for fp in glob.glob(f"{ddir}/*.csv"):
        df=pd.read_csv(fp); df.columns=[c.lower() for c in df.columns]
        for c in ("open","high","low","close"): df[c]=pd.to_numeric(df[c],errors="coerce")
        df=df.dropna(subset=["open","high","low","close"])
        if not len(df): continue
        hi,lo,op,cl=df["high"],df["low"],df["open"],df["close"]
        scale=df[["open","high","low","close"]].abs().max(1).clip(lower=1e-12); tol=RTOL*scale
        bad=(lo-hi>tol)|(df[["open","close"]].max(1)-hi>tol)|(lo-df[["open","close"]].min(1)>tol)
        tot+=len(df); inval+=int(bad.sum())
        with np.errstate(all="ignore"): park=(np.log(hi/lo)**2/LN2x4).where(lambda s: np.isfinite(s))
        ma, mv = park.mean(), park[~bad].mean()
        if bad.sum() and ma>0: shifts.append(abs(mv-ma)/ma)
    s=np.array(shifts); print(group, f"{100*inval/tot:.3f}% invalid",
        f"shift median={np.median(s)*100:.3f}% max={s.max()*100:.3f}%")
```

**Evidence B (GK/RS/YZ already masked):**
```python
import sys, glob, numpy as np, pandas as pd
sys.path.insert(0,"scripts/eda"); import volatility_estimators as VE
RTOL=VE._OHLC_RTOL
def bad_mask(df):
    o,h,lo,c=[pd.to_numeric(df[k],errors="coerce").to_numpy(float) for k in ("open","high","low","close")]
    ok=(np.isfinite([o,h,lo,c]).all(0)&(o>0)&(h>0)&(lo>0)&(c>0)&(h>=lo)
        &(h>=np.maximum(o,c)*(1-RTOL))&(lo<=np.minimum(o,c)*(1+RTOL)))
    return ~ok
for group,ddir in (("HNX","data/raw/prices/hnx_vnstock"),("HOSE","data/raw/prices/hose_vnstock")):
    leaks=0
    for fp in sorted(glob.glob(f"{ddir}/*.csv"))[:400]:
        df=pd.read_csv(fp); df.columns=[x.lower() for x in df.columns]
        if not {"open","high","low","close"}<=set(df.columns): continue
        if "date" in df.columns: df=df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        b=bad_mask(df)
        if not b.sum(): continue
        est=VE.estimators_from_ohlcv(df)
        for e in ("garman_klass","rogers_satchell","yang_zhang"):
            leaks+=int(np.isfinite(est[e].to_numpy()[b]).sum())
    print(group, "finite estimator values on invalid rows (should be 0):", leaks)
```

The invariant to check is that the second script prints `0` for both groups: no estimator produces a finite
value on an invalid-geometry row.
