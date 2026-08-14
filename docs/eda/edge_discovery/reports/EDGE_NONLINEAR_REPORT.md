# Nonlinear edge discovery — gradient-boosting ΔQLIKE + MI-residual

Same leakage-safe harness as the linear test (base = E2; select on validation, judge on held-out test; positivity floor; median + sign-test verdict) but the per-pair predictor is a regularized gradient-boosting regressor, so nonlinear/tail edges are recovered if real.

## Verdict

**STOP — no broad or buildable cross-stock edge: even a nonlinear (gradient-boosting) predictor's val-selected edges are at chance on the held-out test in aggregate. Combined with the linear STOP, no linear OR nonlinear edge is justified as a graph.**

## GB val->test generalization (the decisive test)

- Validation-selected edges (ΔQLIKE_val>0): 378 of 1056.

- Test-positive: **50.0%** (binomial vs 50% p=1.000); held-out-test ΔQLIKE median **-0.00000** (Wilcoxon p=0.6742), mean +0.00016.

- Regime: mean test ΔQLIKE high-vol minus low-vol = -0.00175.

## MI-residual (model-free) — and why its high count is a CONFOUND, not an edge

- Pairs with MI(source(t), E2-residual of target t+5) above the source-permutation null p95: **626 / 1056** (59.3%) — far above the ~5% null rate.

- This does NOT indicate a usable edge. The target residual is de-meaned by MarketPK at time t, but the target lands at t+h and still carries the market factor at t+h, which a leakage-safe base (info at t only) cannot remove. Sources that co-move with the market therefore share that leftover common component. Verified: per-source mean MI correlates 0.77 with |corr(source, MarketPK)|, and MI-above-null is 67% for high-market-correlation sources vs 52% for low — i.e. MI tracks market co-movement, not a pair-specific relation.

## Interpretation

The decisive, confound-free test is the supervised GB val->test generalization above: the gradient-boosting model conditions out the market factor through the base and measures OOS usefulness directly. It lands at ~50% (chance). So even a nonlinear/tail predictor finds no cross-stock edge carrying conditional information beyond the E2 node features. The high model-free MI is a market-leftover artifact — a cautionary example that descriptive dependence must be checked against a supervised, market-conditioned, out-of-sample test.

Honest caveat (not 'zero dependence'): a thin ~10-edge large-cap cluster (Vingroup: BVH->VIC, VIC->VHM, ...) does replicate on test (top-10 val-ranked ~80% test-positive), but it is tail-driven and factor-consistent (source/target market-correlation ~ panel median), i.e. a residual sector/market factor rather than idiosyncratic spillover — and the sibling DM-tested build (2026-08-11_eda_gnn) already showed that graph does not beat HAR (QLIKE p=0.116). So the aggregate is at chance and no buildable graph is justified; the claim is 'no broad/usable cross-stock edge', not literally zero cross-stock dependence.
