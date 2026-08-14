# Adversarial review — nonlinear edge discovery (2026-08-14)

Focused validity layer on `graph_eda/edge_nonlinear.py` (GB ΔQLIKE + MI-residual),
`graph_eda/edge_discovery.py` (predict_fn refactor), `graph_eda/run_edge_nonlinear.py`. The finding
is STOP, so the decisive question was: **is the STOP a false negative, and is the MI-confound claim
correct?**

## Verdict: STOP is ROBUST; MI-confound is CORRECT

| Check | Result |
|---|---|
| GB over-regularized → masks a real edge? | **No.** The production config (depth3/iter100/min_leaf40/l2=1/early-stop) provably recovers a *planted* tail edge (unit test passes). Re-run with a **less-regularized** GB (depth5/iter200/min_leaf15/l2=0): val→test **44.7% test-positive, p=0.018 — BELOW chance**. More capacity → more val-overfit → worse carryover. 50% is the charitable case, not suppression. |
| Exact 50% genuine or degenerate collapse? | **Genuine.** 189/378. Real prediction spread (std 0.0125, max +0.154); **0/378** val-selected pairs are exact-zero ΔQLIKE. corr(dqlike_val, dqlike_test)=0.24 washes to a 50% sign-rate, median −2.7e-6, Wilcoxon p=0.67. |
| MI-confound sound? | **Yes, numbers reproduce exactly.** 59.3% above null-p95; corr(source mean-MI, |corr(source,MarketPK)|)=**0.771**; above-null 67.2% (high-market-corr) vs 51.8% (low). Marginal MI picks up the persistent market factor at t+5 that a leakage-safe base cannot remove; the conditional GB test correctly finds no incremental value. |
| Leakage fabricating signal? | **None material.** Disjoint chronological splits; base/full share the per-pair valid mask; train-only fits + floor. `mi_residual` MI over train+val+test inflates MI (that inflation IS the confound), but no verdict rests on it. |
| Hidden real nonlinear edge? | A thin ~10-edge large-cap/Vingroup tail replicates on test (top-10 ~80% test-positive) but is tail-driven, factor-consistent, and the sibling `2026-08-11_eda_gnn` DM-tested build does not beat HAR (QLIKE p=0.116). Not buildable. |

## Finding applied
- **Honesty phrasing (LOW):** the reviewer noted the conclusion supports "no broad/buildable
  cross-stock edge; aggregate at chance", not literally "zero cross-stock dependence". **Applied** to
  `EDGE_NONLINEAR_REPORT.md` (verdict text + an explicit large-cap-tail caveat).
- No HIGH/MEDIUM code defects. Empty-input guards on `mi_residual` and `_mi_summary` covered by tests;
  ruff clean; diff-cover C0=100% on changed lines.

## Post-review verification
12 tests pass (`.venv default`, py3.14 via pytest-cov env); diff-cover 100%; report regenerated with
the applied caveat.
