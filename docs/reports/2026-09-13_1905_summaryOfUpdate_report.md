# Principled HAR-family feature set — HOSE result (non-inferiority FAIL, but leverage signal found)

Date: 2026-09-13. Baseline: `baselines/2026-09-13_principled_har_features`. Market: HOSE (SP500 via Colab).

## What was built
New isolated baseline replacing the heuristic own-history block (`rq` + `mr_*`) with a citation-backed set:
`[har_daily, har_weekly, har_monthly (Corsi 2009), garman_klass_variance (GK 1980), rogers_satchell_variance
(RS 1991), yang_zhang_n20 (YZ 2000), semi_neg, semi_pos (realized semivariance, Patton-Sheppard 2015)]`. Fixed
Corsi (1,5,22) windows (option A — no GPH/AIC lag selection). GBM(principled) vs GBM(`FM.OWN`), pooled QLIKE +
date-clustered DM + leave-one-out. GK/RS/YZ used from precomputed enriched columns; only new computation =
realized semivariance from `daily_return`.

## HOSE result — principled is WORSE than own-history (fails non-inferiority)
| h | principled QLIKE | own QLIKE | gain % | DM p |
|---|---|---|---|---|
| h1 | 1.6226 | 1.5679 | −3.49 | 0.148 |
| h5 | 1.6585 | 1.6478 | −0.65 | 0.234 |
| h10 | 1.6903 | 1.6842 | −0.36 | 0.018 (sig worse) |
| h22 | 1.7313 | 1.7263 | −0.29 | 0.051 |

The principled set does NOT match own-history: economically worse at h1 (−3.5%), DM-significantly worse at h10.
No overfit (both models `fit=ok`).

## Diagnosis (leave-one-out) — one useful signal, two noise features
- **`semi_neg` (downside semivariance = leverage effect) is genuinely useful:** dropping it HURTS at h1
  (−2.74%) and h22 (−2.06%). This is the one principled addition that carries real signal (bad-volatility
  persistence, Patton-Sheppard 2015).
- **`semi_pos` (upside semivariance) is noise/harmful:** dropping it HELPS (h22 +0.29%, DM p=0.015). Consistent
  with "bad volatility matters more than good volatility".
- **`yang_zhang_n20` is harmful on HOSE:** dropping it HELPS +4.33% at h1 — the windowed YZ estimator is noisy
  on the thin HOSE market (≈45% zero-range days).
- GK/RS contribute modestly (GK helps at h22 −0.85% when dropped); the HAR trio is noisy at h1.

## Conclusion
1. **Own-history (including the heuristic `mr_*`) is hard to beat** — the engineered momentum/mean-reversion
   features earn their place; a "principled/published" replacement is a net loss on HOSE. Reinforces the
   project's standing finding that parsimonious own-history is a strong baseline.
2. **The leverage effect (downside realized semivariance) is the one published signal worth keeping.** The
   right move is to **AUGMENT own-history with `semi_neg`, not replace it** — a promising, cited, single-feature
   addition to test next (FM.OWN + semi_neg, DM vs FM.OWN).
3. `semi_pos` and `yang_zhang_n20` should not be used on HOSE (noise).

## Deliverables
- `baselines/2026-09-13_principled_har_features/` (5 subfolders); `results/gamma_gbm/principled_har_hose.json`;
  Colab notebook `notebooks/principled_har_sp500_colab.ipynb` (SP500 run — expected also ≤ own given HOSE).

## DoD
- Tests: 8 pass (estimators/build_panel/run_har), C0 line = 100%, C1 branch = 100% on changed lines.
- Code review: adversarial, no CRITICAL/MAJOR; 1 MINOR fixed (semivariance NaN-return exclusion) + regression
  test; embargo/overfit constants inline = deliberate protocol parity. See `code_review/`.
- Data-quality: N/A (no data change; reads enriched).
- Performance: seed-averaged HistGBM (OpenMP), batched panel, per-ticker causal semivariance.
- SP500: via Colab (git-centric), not run locally.
