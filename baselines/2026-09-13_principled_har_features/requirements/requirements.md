# Requirements — Principled HAR-family feature set

Full design: `docs/superpowers/specs/2026-09-13-principled-har-features-design.md` (option A, approved).

## Goal
Replace the heuristic own-history block (`rq` + `mr_*`) with a principled, citation-backed feature set and
verify it is **non-inferior** to the current own-history GBM out-of-sample.

## Input
- Enriched per-ticker frames (`data/processed_enriched/<market>/*.csv`) via `FM.load`, providing the
  precomputed columns: `har_daily`, `har_weekly`, `har_monthly` (Corsi 2009 HAR-RV at 1/5/22),
  `garman_klass_variance`, `rogers_satchell_variance`, `yang_zhang_n20` (published range estimators,
  formula-tested), `daily_return`, `parkinson_variance`.
- Target: `y = parkinson_variance.shift(-h)`, h ∈ {1,5,10,22}.

## Output
- `results/gamma_gbm/principled_har_<market>.json`: per horizon — QLIKE(principled, own), DM p (principled vs
  own), leave-one-out per feature, fit diagnostics (train/test QLIKE verdict).

## Feature set
`[har_daily, har_weekly, har_monthly, garman_klass_variance, rogers_satchell_variance, yang_zhang_n20,
semi_neg, semi_pos]` — the only new feature is realized semivariance (`semi_neg`/`semi_pos`, Patton-Sheppard
2015) from `daily_return`. Fixed Corsi windows; NO data-driven lag selection.

## Success / go-no-go
- GO if principled QLIKE is **not significantly worse** than `FM.OWN` under date-clustered DM at all horizons
  (no DM p<0.05 with an economically non-trivial ≥~0.3% loss). With n≈400k, a DM-significant <0.1% gap counts
  as non-inferior. Beating own is a bonus, not required.
- No fit-diagnostics overfit verdict.

## Constraints
- Isolated baseline; no change to shared `FM.OWN` (42 dependents).
- Every named estimator matches its published formula (semivariance formula-tested here; GK/RS/YZ reused from
  the already-tested enriched columns).
- Causal / no look-ahead; HOSE local, SP500 via Colab.
- Tests: C0 line = 100%, C1 branch ≥ 95% on changed lines.
