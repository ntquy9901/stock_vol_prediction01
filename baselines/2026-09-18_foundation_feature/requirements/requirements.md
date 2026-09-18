# Foundation-model feature baseline (Hướng B falsification)

## Objective
Cheaply falsify **Hướng B** (foundation-model prior + GBME): does a **zero-shot** time-series foundation-model
forecast, added as **ONE causal feature** to the champion own-history gamma-GBM (GBME), beat GBME out-of-sample on
QLIKE? The forecast enters as an **additive feature** (NOT the multiplicative anchor, which detonates QLIKE).

Prior belief = **NO-GO** (Brini 2026: zero-shot TimesFM/Chronos lose to Log-HAR under QLIKE). Run honestly; if the
foundation model cannot be installed on this machine, STOP and document — do not fabricate numbers.

## Input / Output
- **Input:** HOSE processed-enriched panel (`data/processed_enriched/hose/*.csv`), own parkinson-variance series;
  crawled VN earnings (`results/gamma_gbm/hose_earnings_combined.parquet`); champion GBME config (OWN-8 + earnings).
- **Foundation model:** Amazon **Chronos-Bolt-tiny** (small, CPU/GPU friendly; avoids the TiRex Windows blocker).
- **Output:** `results/gamma_gbm/foundation_hose_h<h>.json` per horizon h∈{1,5,10,22}, carrying pooled 5-metric
  train/val/test for GBME / GBME+FND / GBME+PLAC, `fit_diagnostics`, Stage-0 (zero-shot vs HAR), Stage-1 DM, placebo,
  per-fold, spike robustness; plus a forecast cache parquet `foundation_cache_hose.parquet`.

## Mechanism
- **Stage 0 (zero-shot vs HAR):** for each stock, causal zero-shot forecast (context ≤ t) of h-step-ahead variance;
  score QLIKE + date-clustered DM vs HAR per horizon. Record whether it even matches HAR.
- **Stage 1 (additive feature):** add the zero-shot forecast (+ its quantile spread) as extra causal GBME columns.
  Compare GBME vs GBME+FND (date-clustered DM, spike-robust) plus a **wrong-ticker placebo** (GBME+PLAC) that must
  NOT reproduce any gain.

## Success criteria (pre-registered kill)
GBME+FND beats GBME DM-significantly (gain>0 AND date-clustered DM p<0.05) at **≥2 horizons**, the win survives the
regime-spike exclusion (COVID / 2022 / Apr-2025), AND the placebo does not beat GBME → otherwise **NO-GO**.

## Go / No-Go
- **Go (surprising):** ≥2 horizons beat, spike-robust, placebo negative.
- **No-Go (expected):** foundation feature does not beat the champion, or the placebo reproduces the "gain".
- **Blocked:** foundation model not installable → STOP + document the install error; no fabricated numbers.

## Constraints
- Zero-shot inference strictly causal (context ≤ t); frozen model, no fine-tuning, no leakage.
- Forecasts cached to parquet (computed once); batched GPU inference (never batch=1); tunable constants only in
  `foundation_config.py`.
- Hard isolation: read-only imports from `scripts/eda` (full_matrix, stage1) + shared metrics/stats/overfit_check.
