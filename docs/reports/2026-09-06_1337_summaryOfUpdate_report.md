# Summary of update — limit-lock hurdle baseline (shock-detection probe #2)

## What changed
Added `baselines/2026-09-06_limitlock_hurdle/` (§3.F, 5 subfolders): a two-part hurdle on the QLIKE champion
HAR-X that tries to stop QLIKE exploding on limit-lock / zero-range days (the days that dominate VN QLIKE,
e.g. VN30 2025-04-10). Motivated by the shock-detection empirical analysis: the days that break QLIKE are not
high-vol days but limit-lock days where Parkinson variance collapses to ~0 and models over-forecast.

## Design
- Causal limit-lock features (`limitlock_features.py`): `limit_down_streak`, `recent_lock_freq`,
  `near_limit_freq` (trailing window, min_periods=1, prefix-invariance unit-tested).
- Two-part model (`run_limitlock_hurdle.py`): (A) logistic classifier predicts P(zero_range at t+h) from the
  features (train folds only); (B) HAR-X OLS variance forecast. On test cells with P(lock) >= threshold,
  override the forecast with the shared positivity floor. Thresholds swept {0.5, 0.7, 0.9}.
- Added an **oracle variant** (override on the TRUE lock flag) as the ceiling test, to separate "the idea is
  weak" from "the classifier is weak". Same QLIKE floor + positivity floor across compared models (H2).
- CPU-only (OLS + sklearn logistic); GPU untouched (ran alongside the lookback=10 GPU re-run).

## Result — NO-GO (two independent reasons)
1. **The classifier cannot predict VN limit-locks:** 0 of 42 (VN30) / 0 of 153 (VN100) true lock days caught at
   any threshold; max P(lock) on a true lock cell = 0.05. The practical hurdle is a no-op (QLIKE identical to
   plain HAR-X to 4 dp; `qlike_robust` byte-identical → zero false-positive harm on normal days).
2. **Even an oracle (perfect classifier) barely helps and is not DM-significant:** overriding on the true lock
   flag cuts pooled QLIKE only ~3–4% (VN30 h1 0.4967→0.4759) at DM p 0.11–0.23. The positivity floor caps
   lock-day QLIKE at ~5 (cannot reach 0), so the ceiling of the whole idea is small.

| market/h | HAR-X QLIKE | oracle QLIKE | lock-day QLIKE HAR-X→oracle | locks caught | oracle DM p |
|---|---|---|---|---|---|
| vn30 h1 | 0.4967 | 0.4759 | 10.34 → 4.96 | 0/42 | 0.194 |
| vn30 h5 | 0.5929 | 0.5765 | 9.16 → 4.96 | 0/42 | 0.204 |
| vn100 h1 | 0.5006 | 0.4846 | 10.01 → 5.17 | 0/153 | 0.110 |

### 2025-04-10 VN30 case (concrete)
30 of 31 eligible stocks hit zero-range. HAR-X over-forecasts each (~10.6 QLIKE/cell). The classifier assigned
P(lock) 0.003–0.050 → caught 0. Root cause: the lock was a limit-**up** rebound preceded by a limit-**down**
streak — an exogenous tariff-shock event, unpredictable from autocorrelated limit-lock history. Oracle override
halves the per-cell QLIKE (~10.6→4.96), which is the ceiling.

## Interpretation
This is the second independent shock-detection NO-GO (after `regime_features`). Both confirm the empirical
finding: VN volatility shocks are exogenous-onset (tariff/policy news) and not forecastable from price/volume
history; and the limit-lock day that breaks QLIKE is unpredictable from lock-history features. The defensible
fixes remain (a) robust QLIKE reporting (delivered) and, for onset detection, (b) EXOGENOUS early-warning data
(US overnight return, VIX, news) — not an architecture change on the same price features. Extension to deep
models is not recommended: the blocker is the unpredictable target + the floor ceiling, a property of the
problem, not of the linear probe.

## Tests + coverage
`pytest baselines/2026-09-06_limitlock_hurdle/test/ -q` → 18 passed. C0 = 100% on all non-`pragma` lines;
`ruff --select F` clean. Includes a real-data smoke on the VN30 FPT file.

## Code review
Adversarial review (coordinator): features causal (feature at t → target lock at t+h, train-only fit),
hard-isolated (delivered modules read-only), shared floor (H2), oracle correctly labeled as a ceiling (not a
leakage claim), vectorised (one logistic fit per fold). No critical/major findings. See
`code_review/code_review_2026-09-06.md`.

## Data-quality gate
N/A (no data change) — reuses cleaned VN30/VN100 enriched panels + existing `zero_range_flag`; features derived
in-memory.

## Overfit-evidence gate
Skipped by design: models are OLS/logistic (`HAR-X`, `HAR-X+hurdle@...`), none match the learned-model patterns
and no per-seed/masked schema, so `check_overfit_evidence` treats these as non-training JSONs (same as
`regime_features`). No fabricated evidence added.

## Risks / follow-ups
- The onset-detection direction now points squarely at exogenous data (US/VIX/news) — a larger, separate effort.
- Paper: record this + the regime NO-GO as tried-and-rejected shock-detection extensions; the honest story is
  robust QLIKE, not a shock predictor.

## DoD checklist
- [x] Code matches request, no unrelated refactor, hard-isolated
- [x] Tests + C0 100% on changed lines
- [x] Code review done, no critical/major
- [x] Summary report (this file)
- [ ] Push (after commit)
