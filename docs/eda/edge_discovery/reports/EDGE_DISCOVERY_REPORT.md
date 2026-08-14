# Edge-Discovery EDA — supervised OOS ΔQLIKE directed edges

Base = E2 node features (HAR + MarketPK + volume_zscore_20). An edge i->j helps only if adding source i's PK history LOWERS target j's out-of-sample QLIKE beyond that base (positive ΔQLIKE). h=5, positivity-floored, leakage-safe: coefficients fit on TRAIN, edges SELECTED on VALIDATION only, generalization judged on the untouched TEST.

## Verdict

**STOP — validation-selected edges do NOT generalize to the held-out test (test-positive rate at/below chance, mean test ΔQLIKE <= 0). No edge definition recovers conditional cross-stock signal beyond E2 even on QLIKE; a cross-stock GNN is not justified.**

## Decisive test — leakage-safe val->test generalization

- Validation-selected edges (ΔQLIKE_val>0, no test peek): 449 of 1056 directed pairs.

- Of those, **test-positive: 45.9%** (binomial vs 50% chance p=0.089); held-out-test ΔQLIKE of the selected set: mean **-0.01241**, median **-0.00021** (Wilcoxon p=0.0006). Verdict gates on the robust median, not the tail-sensitive mean.

- Reading: 46% ~ chance and mean <= 0 means selection does not carry over — the edges that look helpful on validation are noise, not conditional signal.

## Context — test-peeking counts (NOT the verdict basis)

- Sign-stable (ΔQLIKE>0 on val AND test): 206; FDR-'significant' helpful (q<0.05 AND test>0): 47. These use the test set in selection, so with ~1000 pairs x ~14k obs a few dozen cross the line by chance; they do NOT survive the honest val->test split above.

- Regime: mean test ΔQLIKE high-vol minus low-vol = +0.02340 (more help under stress).

- Market-residual lead-lag L(5): mean |off-diag| = 0.0615, max = 0.3297.

## Top-15 directed edges by test ΔQLIKE (test-ranked; note most have ΔQLIKE_val<=0)

| source -> target | ΔQLIKE val | ΔQLIKE test | p_test | fdr_q | stable |
|---|---|---|---|---|---|
| VHM -> PLX | +0.00155 | +0.05462 | 0.0032 | 0.0345 | True |
| BCM -> VCB | -0.02939 | +0.03875 | 0.0874 | 0.2828 | False |
| VIC -> VRE | +0.00860 | +0.03421 | 0.0004 | 0.0083 | True |
| TCB -> PLX | -0.00357 | +0.03341 | 0.0061 | 0.0562 | False |
| PLX -> TCB | -0.01576 | +0.02827 | 0.0097 | 0.0761 | False |
| HPG -> PLX | -0.00918 | +0.02790 | 0.0006 | 0.0109 | False |
| MWG -> PLX | -0.00323 | +0.02685 | 0.0018 | 0.0216 | False |
| VJC -> BVH | +0.00598 | +0.02591 | 0.0457 | 0.1952 | True |
| BCM -> BID | -0.10528 | +0.02381 | 0.0712 | 0.2551 | False |
| VNM -> VIB | +0.01373 | +0.02263 | 0.0197 | 0.1211 | True |
| VIB -> GAS | -0.05982 | +0.02232 | 0.0011 | 0.0168 | False |
| VJC -> GAS | -0.00490 | +0.02147 | 0.0093 | 0.0746 | False |
| PLX -> VIB | -0.00335 | +0.02105 | 0.1209 | 0.3436 | False |
| PDR -> VIC | +0.00878 | +0.02097 | 0.0000 | 0.0000 | True |
| VJC -> VHM | +0.00056 | +0.02095 | 0.0572 | 0.2241 | True |

## Caveats & robustness

- The per-edge `p_test`/`fdr_q` are paired-t on daily QLIKE differences with an overlapping 5-day horizon (serially correlated), so they are anti-conservative and shown for context only. The verdict rests on the sign-count val->test binomial across PAIRS (not obs), which is unaffected by that autocorrelation.

- Source feature = raw PK_i(t). Because the base already contains MarketPK and an intercept, adding raw PK_i or its market-residual spans the same column space -> the ridge already orthogonalizes the edge against the market factor (residualizing is a no-op, verified). So this IS the market-adjusted edge test.

## Interpretation

ΔQLIKE>0 means the source lowers the target's QLIKE out of sample after the market factor and the target's own node features are already known — i.e. genuine conditional cross-stock information. If the FDR-controlled stable set is empty/tiny, the correlation the earlier G1/E3 graphs used was mostly the market factor, and no edge definition recovers conditional signal — a stronger stop-GNN result than the RMSE-only ranking gave. If a set exists, Phase B builds exactly those directed edges on the E2 pooled backbone and DM-tests vs E2/HAR.
