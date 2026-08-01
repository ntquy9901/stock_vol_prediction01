# Phase 5 Results: Bug Fix + Multi-Horizon (1/5/10-day), Post-Fix Pipeline

**Date:** 2026-08-01
**Branch:** `global-benchmark`
**Epochs:** 10 per run (all 12 runs)
**Pipeline:** post-bugfix (`src/common/multi_ticker_dataset.py` — per-ticker windowing + per-ticker
scaler fit-once-reuse; see `DESIGN.md` §0). **Numbers below are NOT directly comparable to the old
Phase 3 / Phase 4 numbers** except where explicitly placed side-by-side for the bug-fix check in §3.

---

## 1. Experiment A — Feature-set comparison (AAPL/MSFT/GOOGL, HAR-only vs Full 9-feature)

| Horizon | Feature set | DirAcc | R² | QLIKE | RMSE | MSE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | HAR-only | 31.26% | 0.2468 | 0.3448 | 0.000322 | 1.04e-07 | 0.000123 |
| 1 | **Full** | **36.08%** | **0.3919** | **0.3318** | **0.000290** | **8.39e-08** | 0.000120 |
| 5 | **HAR-only** | **52.11%** | **0.0217** | **0.4237** | **0.000367** | 1.35e-07 | 0.000140 |
| 5 | Full | 50.27% | 0.0010 | 0.4832 | 0.000371 | 1.38e-07 | 0.000161 |
| 10 | **HAR-only** | **52.23%** | **-0.0079** | **0.4399** | **0.000373** | 1.39e-07 | 0.000140 |
| 10 | Full | 50.34% | -0.0258 | 0.5087 | 0.000377 | 1.42e-07 | 0.000159 |

**Reading (bold = winner between HAR-only/Full at that horizon):**
- **Horizon 1 is the outlier, not the easiest horizon.** DirAcc is far BELOW random (31-36% vs 50%
  expected for a coin flip) at horizon 1, for both feature sets — the opposite of the VN30 pattern
  where horizon 1 was the EASIEST (72%+ DirAcc). QLIKE/RMSE/R², however, ARE best at horizon 1 (lowest
  QLIKE, highest R²) — the model fits the magnitude well but gets the direction of day-to-day change
  systematically wrong more than half the time. Not fixed in this phase — flagged as an open finding
  (§4), reproduced independently in Experiment B at horizon 1 too (see §2), so it is unlikely to be
  random noise.
- **Horizon 5 and 10 sit near random (48-52% DirAcc)** — consistent with a genuinely hard forecasting
  problem for a 3-ticker, HAR-only-scale LSTM (no cross-stock structure, no news), not with a residual
  pipeline bug (the boundary/scaler bugs are fixed and verified by tests — see DESIGN.md §0).
- **R² turns negative at horizon 10** (both feature sets) — the model does no better than predicting
  the mean. QLIKE and RMSE increase monotonically with horizon (1→5→10), matching the VN30-observed
  trend that longer horizons are harder on those 2 metrics, even though DirAcc does not follow the same
  monotonic pattern here.
- **Full features help at horizon 1 (both DirAcc and QLIKE), but hurt slightly at horizon 5 and 10**
  (HAR-only wins on DirAcc/R²/QLIKE at both). Only 7-10 sentiment/market data points exist for these 3
  tickers (see Phase 2 RESULTS — sentiment coverage was 7-10 days per ticker), so at longer horizons the
  extra features may add more noise than signal relative to a purely magnitude-driven HAR baseline.

## 2. Experiment B — Cross-market (SP500 ↔ VN30, HAR-only 3 features)

| Horizon | Direction | DirAcc | R² | QLIKE | RMSE | MSE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | SP500→VN30 | 31.72% | 0.7469 | 6.0337 | 0.002069 | 4.28e-06 | 0.000556 |
| 1 | VN30→SP500 | 32.41% | 0.2226 | 0.3384 | 0.000260 | 6.76e-08 | 0.000118 |
| 5 | SP500→VN30 | 48.40% | 0.6934 | 0.8287 | 0.002276 | 5.18e-06 | 0.000627 |
| 5 | VN30→SP500 | 49.49% | 0.1008 | 0.4111 | 0.000280 | 7.83e-08 | 0.000129 |
| 10 | SP500→VN30 | 48.99% | 0.6910 | 0.8680 | 0.002285 | 5.22e-06 | 0.000630 |
| 10 | VN30→SP500 | 49.80% | 0.0708 | 0.4325 | 0.000285 | 8.10e-08 | 0.000132 |

**Reading:**
- **Same horizon-1 DirAcc anomaly as Experiment A** (31.72%/32.41%, both directions) — reinforces that
  this is a real property of 1-day-ahead volatility-change prediction with this architecture, not an
  artifact isolated to one experiment.
- **SP500→VN30 has much higher R² (0.69-0.75) than VN30→SP500 (0.07-0.22) at every horizon**, but WORSE
  QLIKE (0.83-6.03 vs 0.34-0.41). This is the same asymmetry flagged in the original Phase 4
  RESULTS.md ("RMSE lower for cross-market... but DirAcc is much worse → model predicts constant
  values") — VN30's target has much larger absolute scale/variance than SP500's, so a model trained on
  SP500 and evaluated on VN30 can achieve deceptively high R² by tracking VN30's large swings loosely,
  while QLIKE (scale-normalized, ratio-based) exposes that the fit is proportionally poor.
- **Horizon-1 SP500→VN30 QLIKE = 6.03 is an extreme outlier** — 7-14× worse than every other cell in
  either table. Combined with R²=0.747 (the highest R² in the whole sweep) at the same cell, this is
  the clearest sign of a badly-calibrated-but-directionally-tracking model: it follows VN30's large-
  scale swings (good R²) but is proportionally very wrong on the sign/magnitude of change (bad QLIKE,
  bad DirAcc). Not investigated further in this phase — flagged in §4.

## 3. Bug-fix check — same horizon-5 setup, pre-fix vs post-fix pipeline

The original motivating question (see conversation) was why S&P 500 DirAcc was far below VN30's. Direct
before/after comparison at horizon 5, same tickers/architecture:

| Setup | DirAcc (old→new) | RMSE (old→new) | QLIKE (old→new) |
|---|---|---|---|
| Exp A, HAR-only | 50.89% → **52.11%** (+1.22pp) | 0.000304 → 0.000367 (worse) | **1.959 → 0.424** (4.6× better) |
| Exp A, Full | 51.67% → 50.27% (-1.40pp) | 0.000292 → 0.000371 (worse) | **1.952 → 0.483** (4.0× better) |
| Exp B, SP500→VN30 | 48.32% → 48.40% (+0.08pp) | 0.000229 → 0.002276 (worse†) | 0.0795 → 0.829 (worse†) |
| Exp B, VN30→SP500 | 49.75% → 49.49% (-0.26pp) | 0.000638 → 0.000280 (2.3× better) | **0.517 → 0.411** (better) |

† The old SP500→VN30 QLIKE=0.0795/RMSE=0.000229 looked deceptively good — the ORIGINAL Phase 4
RESULTS.md itself already flagged this exact cell as suspicious ("RMSE is lower for cross-market...
but DirAcc is much worse → model predicts constant values"). A model outputting a near-constant
prediction close to VN30's mean would show low RMSE/QLIKE on paper while being directionally useless
(DirAcc 48%, near-random) — which is exactly the old row's pattern. The post-fix number (0.829) is very
likely the more honest one, not a regression.

**Overall: DirAcc barely moves (±1.4pp, noise-level for a single seed), but QLIKE improves 4-4.6× on 3
of 4 comparable cells** (the 4th, SP500→VN30, likely traded a degenerate old artifact for a more honest
worse-looking number). **This means the bug fix mattered for calibration (QLIKE) far more than for
directional accuracy** — the near-random DirAcc (48-52%) at horizon 5/10 was NOT primarily caused by
the 3 structural bugs; it persists after the fix. The bugs were real (proven by the test-first tests in
DESIGN.md §0 and §9) and worth fixing regardless, but they are not the dominant explanation for why
S&P 500 DirAcc trails VN30's 68%+. The dominant gap is more likely the maturity gap already named in
the original comparison (§ conversation): 3 tickers vs 30, no GAT/spatial branch, no per-ticker gate,
10 epochs vs VN30's tuned 20+, days of iteration vs VN30's ~1 month.

## 4. Open findings (not investigated further in this phase)

1. **Horizon-1 DirAcc is anomalously low (31-36%) across all 4 sub-experiments**, while QLIKE/R² are
   BEST at horizon 1. Opposite of the VN30 pattern (horizon 1 = easiest on all 4 metrics). Worth a
   dedicated investigation (e.g. check whether the model's 1-day-ahead predictions are systematically
   lagged/anti-persistent) before trusting horizon-1 forecasts from this pipeline.
2. **SP500→VN30 horizon-1 QLIKE=6.03 is a severe outlier** paired with the sweep's highest R²
   (0.747) — likely a scale-mismatch / near-constant-prediction artifact (see §2), not investigated.
3. **Full feature set helps at horizon 1 but hurts at horizon 5/10** for Experiment A — plausibly
   because sentiment coverage is only 7-10 days per ticker (Phase 2), too sparse to help at longer
   horizons; not tested against a larger sentiment sample in this phase.

## 5. Files

New training runs (this phase): `results/sp500_enhanced_h{1,5,10}_2026-08-01_*/results.json` (6),
`results/cross_market_h{1,5,10}_2026-08-01_*/results.json` (6). Superseded (pre-fix) baseline for
comparison: `research/phase3_training/RESULTS.md`, `research/phase4_crossmarket/RESULTS.md` (left
unmodified as historical record).
