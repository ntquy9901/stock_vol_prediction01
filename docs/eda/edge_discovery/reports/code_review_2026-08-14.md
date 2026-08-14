# Adversarial review — edge-discovery EDA (2026-08-14)

Two focused layers on `graph_eda/edge_discovery.py` + `graph_eda/run_edge_discovery.py`. The finding
is a STOP (no build), so the decisive review question was: **is the STOP a false negative from a
methodology bug?**

## Statistical-validity / false-negative layer — STOP is ROBUST
Re-ran the full 1056-pair pipeline under 8 alternative methodologies; none flips the verdict:
| Variant | Result |
|---|---|
| Positivity floor train-p1 (default) | 45.9% test-pos, median <0 (Wilcoxon p≈6e-4) |
| No floor (eps only) | mean explodes (−13.5, tail catastrophe), median still <0 (p≈2e-3) |
| Trimmed means 1/5/10% | stay negative; frac>0 ≈ 0.45 at every trim |
| Source = market-residual ε_i vs raw PK_i | **numerically identical** (base MarketPK+intercept ⇒ ridge orthogonalizes; residualizing is a no-op) |
| Ridge λ=1.0 vs 1e-4 | identical (45.6–45.8%) |
| Selector = val paired-t p<0.05 | 42 edges, 47.6% test-pos, binom p=0.878 (dead chance) |
| Median / Wilcoxon | selected edges *significantly hurt* OOS (p≈6e-4) |
| Regime high-vol | median negative, Wilcoxon p≈0.019 (hurt more under stress) |

Conclusion: no leakage fabricates the residual 45.9%; no bug suppresses a real edge. Under robust
statistics the mean-based STOP *understates* the effect. A cross-stock edge beyond E2 is not
justified.

## Leakage / edge-case layer — no leakage; 3 defects fixed
Verified correct: train-only ridge β + standardization + floor + regime thresholds; strict-future
target `shift(-horizon)`; base/full share one per-pair valid mask (like-for-like); per-obs ΔQLIKE
aligned (identical NaN positions); constant-column safe (`sd==0→1`); pk_var inf sanitized.

Fixes applied:
| # | Sev | Finding | Fix |
|---|---|---|---|
| 1 | MEDIUM | `_summarise`/`_heatmap` crash (KeyError / vmax=NaN) on an empty edge table (reachable on a <120-row panel) | early-return STOP summary when empty; finite-guard vmax; unit test added |
| 2 | MEDIUM | per-edge `p_test`/`fdr_q` anti-conservative under overlapping 5-day horizon (serial correlation) | flagged in report as context-only; verdict uses the pair-level sign binomial (unaffected) |
| 3 | LOW/MED | verdict "helps on average" gate used a tail-sensitive mean | switched to robust **median** + report Wilcoxon; a single QLIKE-spike pair can no longer drive the verdict |

Minor (no action): `n_test` counts all valid rows vs positive-only ΔQLIKE (reported only); `_fit`
refits base twice (redundant, not incorrect); empty-regime nanmean warnings (cosmetic).

## Post-fix verification
7 tests pass; ruff clean; diff-cover C0=100% on changed lines; real run reproduces STOP
(45.9% test-positive, median <0). No HIGH findings; all MEDIUM fixed.
