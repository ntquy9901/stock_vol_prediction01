# VolGA spillover-edge sparsity EDA: is density the QLIKE lever?

Date: 2026-09-06
Scope: analysis-only (no training, CPU). Edge under study: `directed_vol2pk_hmatched`
(`baselines/2026-09-05_edge_horizon_matched/code/run_edge_hmatched.py:57-104`).
Inputs: `data/processed_enriched/{vn30,vn100,sp500_clean}/*.csv` (train windows only) and the 12
delivered result JSONs `results/edge_hmatched/edgehm_{market}_h{1,5,10,22}.json`.
Reproducible artifacts: `scripts/eda/volga_edge_sparsity_eda.py` (JSON:
`docs/reports/2026-09-06_volga_edge_sparsity_eda.json`), `scripts/eda/volga_edge_sparsity_plots.py`
(HTML: `docs/reports/2026-09-06_volga_edge_sparsity_eda.html`).

## Bottom line

**Density is not the lever.** Across the 12 (market x horizon) configs, edge density has essentially
zero rank-correlation with whether the graph beats the no-graph LSTM (Spearman **+0.19** density vs
`QLIKE(LSTM)-QLIKE(VolGA)`; **-0.15** density vs the VolGA-vs-LSTM DM p-value). What *does* track graph
benefit is the **strength of the spillover signal** (Spearman **+0.73** signal-excess vs benefit;
**-0.50** vs DM p). The densest config (VN30 h1, 12.8%) shows the *smallest, non-significant* benefit,
while the *sparsest* h1 config (SP500 1.04%) shows the *largest, most significant* benefit. The
significance screen is doing its job: the volume->vol(t+h) correlation signal is statistically real at
short horizons (6-19x the null rate) but the *effect sizes are tiny* (median |r| ~ 0.02-0.07,
max |r| < 0.25 everywhere). The recommendation is a narrow, low-ceiling one (raise Top-K only on
SP500) and, per the strong 2026-08-29 prior, denser/alternative edges are not expected to move QLIKE.

## 1. Sparsity decomposition -- Bonferroni floor vs Top-K cap

Per-target counts, averaged over the 7 walk-forward train folds (`sig` = sources clearing the
Bonferroni floor *before* Top-K; `kept` = after Top-K=5):

| config | N | density | sig/tgt | kept/tgt | self-loop-only | Top-K binding (sig>5) | Bonferroni binding (0<sig<=5) |
|---|--:|--:|--:|--:|--:|--:|--:|
| vn30 h1 | 33 | 12.80% | 7.76 | 4.10 | 10.8% | 71.4% | 17.7% |
| vn30 h5 | 33 | 3.44% | 1.14 | 1.10 | 47.2% | 3.5% | 49.4% |
| vn30 h10 | 33 | 2.91% | 0.93 | 0.93 | 48.9% | 0.0% | 51.1% |
| vn30 h22 | 33 | 0.78% | 0.25 | 0.25 | 79.2% | 0.0% | 20.8% |
| vn100 h1 | 102 | 3.52% | 7.60 | 3.55 | 13.0% | 49.7% | 37.3% |
| vn100 h5 | 102 | 1.05% | 1.15 | 1.06 | 47.9% | 2.2% | 49.9% |
| vn100 h10 | 102 | 0.96% | 1.01 | 0.97 | 51.5% | 2.5% | 45.9% |
| vn100 h22 | 102 | 0.47% | 0.48 | 0.48 | 65.0% | 0.0% | 35.0% |
| sp500 h1 | 480 | 1.04% | **318.2** | 5.00 | 0.0% | **100.0%** | 0.0% |
| sp500 h5 | 480 | 1.04% | 88.5 | 4.96 | 0.0% | 98.5% | 1.5% |
| sp500 h10 | 480 | 1.02% | 78.6 | 4.89 | 0.7% | 95.0% | 4.3% |
| sp500 h22 | 480 | 0.47% | 4.08 | 2.23 | 25.3% | 20.7% | 54.0% |

Two distinct regimes:

- **Short horizon (h1) -> Top-K is the binding constraint.** On SP500 h1, **318 of 479** candidate
  sources per target clear the Bonferroni floor, but Top-K=5 keeps only 5 -> density is pinned at
  5/479 = **1.04%** by the cap, *not* by the significance screen. VN h1 is a milder mix (~50-71% of
  targets have more than 5 significant sources). **Density here is a function of N and Top-K, not of
  how much signal exists** (this is why the big-N SP500 looks "sparse" at 1% while the small-N VN30
  looks "dense" at 13% -- 5/32).
- **Long horizon (h5/h10/h22) -> Bonferroni floor + self-loop fallback dominate.** `sig/tgt` collapses
  to ~1 or below, `kept ~= sig` (Top-K never bites), and the self-loop-only fraction rises to
  47-79%. At VN h22, 65-79% of targets keep only their self-loop -> **VolGA = LSTM there by
  construction** (Section 5).

## 2. Correlation structure -- is the screen pruning signal or noise?

Analytic test on the pairwise-complete correlations (z = |r|*sqrt(n_pairs); under the null z~N(0,1),
so 5% of valid pairs exceed |z|>1.96 by chance). "Excess" = observed count / null expectation:

| config | valid pairs | n(|z|>1.96) | expected(null) | **excess** | median|r| | p95|r| | max|r| |
|---|--:|--:|--:|--:|--:|--:|--:|
| vn30 h1 | 1056 | 493 | 53 | **9.4x** | 0.041 | 0.100 | 0.155 |
| vn30 h5 | 1056 | 205 | 53 | 3.9x | 0.024 | 0.065 | 0.119 |
| vn30 h10 | 1056 | 173 | 53 | 3.3x | 0.022 | 0.065 | 0.137 |
| vn30 h22 | 1056 | 88 | 53 | 1.7x | 0.018 | 0.053 | 0.090 |
| vn100 h1 | 10302 | 3177 | 515 | **6.2x** | 0.027 | 0.074 | 0.146 |
| vn100 h5 | 10302 | 1428 | 515 | 2.8x | 0.018 | 0.054 | 0.122 |
| vn100 h10 | 10302 | 1364 | 515 | 2.6x | 0.018 | 0.055 | 0.138 |
| vn100 h22 | 10302 | 987 | 515 | 1.9x | 0.016 | 0.052 | 0.118 |
| sp500 h1 | 229920 | 217518 | 11496 | **18.9x** | 0.065 | 0.110 | 0.222 |
| sp500 h5 | 229920 | 155851 | 11496 | 13.6x | 0.037 | 0.072 | 0.166 |
| sp500 h10 | 229920 | 147796 | 11496 | 12.9x | 0.035 | 0.071 | 0.173 |
| sp500 h22 | 229920 | 45019 | 11496 | 3.9x | 0.015 | 0.044 | 0.143 |

Findings:

- **The signal is real, not noise.** At h1 the number of significant volume->vol(t+h) correlations is
  6-19x what pure noise would produce; the p-value mass has a genuine spike near zero. So the
  Bonferroni screen is *not* just removing chance correlations -- there is true cross-sectional
  spillover, strongest on SP500.
- **But the effect sizes are economically tiny.** Even the strongest edges are max |r| < 0.25 and the
  median significant edge is |r| ~ 0.03-0.07. A single day-t volume shock at another stock explains a
  fraction of a percent of a target's variance at t+h. This is the ceiling on any graph gain.
- **Signal decays sharply with horizon:** excess falls from 6-19x (h1) to 1.7-3.9x (h22). This tracks
  exactly where VolGA stops helping (Section 3), and matches the design comment that the lead-lag
  signal is at noise level by h22 (`run_edge_hmatched.py:14-15`).

## 3. Density vs performance (the paradox, resolved)

Spearman rank-correlations across the 12 configs (benefit = `QLIKE(LSTM) - QLIKE(VolGA)`, >0 = graph
helps; DM p from `dm_date_clustered.VolGA_vs_LSTM.qlike.p_value`):

| relationship | Spearman |
|---|--:|
| **density** vs benefit-vs-LSTM | **+0.19** (~0) |
| **density** vs DM p-value (VolGA vs LSTM) | **-0.15** (~0) |
| horizon vs benefit-vs-LSTM | -0.59 |
| **signal-excess** vs benefit-vs-LSTM | **+0.73** |
| **signal-excess** vs DM p-value | **-0.50** |

The h1 slice makes the paradox concrete -- benefit *decreases* as density *increases*:

| config | density | signal excess | benefit-vs-LSTM | DM p (vs LSTM) |
|---|--:|--:|--:|--:|
| sp500 h1 | **1.04%** | 18.9x | **+0.083** | 0.000 |
| vn100 h1 | 3.52% | 6.2x | +0.028 | 0.006 |
| vn30 h1 | **12.80%** | 9.4x | +0.005 | 0.201 (n.s.) |

The four configs where VolGA *significantly* beats LSTM (DM p<0.05) are SP500 h1/h10/h22 and VN100 h1
-- three of them among the *sparsest* densities in the study. Density is set by N and the Top-K cap;
graph benefit is set by signal strength (excess) and horizon. They are decoupled. (A `+0.68` Spearman
of density vs benefit-vs-HAR-X is a market confound: SP500 has both low density and HAR-X strongly
dominating the deep models -- it is not a density effect.)

## 4. Denser-edge candidates (EDA-only edge counts)

Density (%) that each variant would produce (Bonferroni/Top-K counts before training):

| config | delivered | alpha=.10 | alpha=.20 | Top-K=10 | Top-K=20 | BH-FDR .05 | BH-FDR .10 | vol->vol | \|ret\|->vol |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| vn30 h1 | 12.80 | 13.30 | 13.79 | 21.28 | 24.24 | 39.76 | 45.64 | 15.62 | 15.62 |
| vn100 h1 | 3.52 | 3.84 | 4.12 | 5.53 | 7.28 | 18.19 | 24.62 | 4.95 | 4.95 |
| sp500 h1 | 1.04 | 1.04 | 1.04 | 2.09 | 4.17 | **94.22** | 96.22 | 1.04 | 1.04 |
| sp500 h22 | 0.47 | 0.59 | 0.72 | 0.64 | 0.78 | 5.63 | 9.61 | 1.04 | 1.04 |

(Full 12-config table in the JSON.) Interpretation:

- **FDR / Benjamini-Hochberg** is far less conservative -- on SP500 h1 it "keeps" 94% of pairs. But
  those added edges are exactly the tiny-|r| tail (Section 2): with N=480 even r~0.05 is
  "significant". This is adding near-noise edges by the thousand, the opposite of what the delivered
  screen is for. High risk, matches the 2026-08-29 corr+lift near-empty/over-full failure mode.
- **Relaxed alpha (0.10, 0.20)** barely moves density where Top-K already binds (SP500, VN h1); it
  only adds edges at long horizon, i.e. exactly the low-signal regime where the graph does not help.
- **Higher Top-K (10, 20)** is the only variant that adds *top-ranked* (largest-|r|) spillover edges to
  configs that already benefit: SP500 h1 goes 1.04% -> 2.09% (K=10) -> 4.17% (K=20), all from the
  318-source pool that currently loses 313 real-but-capped sources. This is the one denser variant
  whose added edges are the *strongest*, not the weakest.
- **vol->vol and |return|->vol** edges are also Top-K-capped (density = 5/N), i.e. plenty of sources
  pass -- volatility co-moves strongly -- but they encode contemporaneous comovement, which the
  per-node LSTM largely already captures, and the 2026-08-29 correlation-vol2pk / DY-spillover probes
  found no OOS lift from such edges.

## 5. Thin-market check (long-horizon self-loop fallback)

At the sparsest configs the graph collapses to no-graph for most targets: self-loop-only fraction is
**65% (VN100 h22)**, **79% (VN30 h22)**, **25% (SP500 h22)**. Where the fallback dominates, VolGA is
LSTM plus a handful of self-loops -> the observed VolGA~=LSTM QLIKE at long h is *by construction*, not
a modelling failure. This is the intended behaviour of the significance floor
(`run_edge_hmatched.py:12-16`): it removes the edge exactly where the lead-lag signal is at noise
level, which is why VolGA does not *hurt* at long h (unlike the fixed lag-1 edge it replaced).

## Prior work this builds on (do not re-derive)

- 2026-08-29 edge-construction robustness (`project_edge_construction_robustness_2026-08-29`;
  reports `2026-08-29_{corrlift,dy_spillover,graphwavenet,learned_graph_mtgnn,sector_gat_scaleup}.md`,
  `2026-08-16_2330_p5_glasso_edge_vs_vol2pk_report.md`): **six** structurally different edges
  (correlation-vol2pk, sector-ICB, MTGNN-learned, DY-spillover, Graph-WaveNet-adaptive, corr+lift) all
  failed to beat no-graph LSTM on HNX h1; some were significantly worse; the corr+lift graph was
  near-empty on thin returns. Denser/alternative edges therefore carry a *strong negative prior*.
- The current sparse horizon-matched edge *does* work at h1 (VN100 DM p=0.006 vs LSTM, p=0.047 vs
  HAR-X; SP500 all h vs LSTM) -- i.e. keeping only genuine, top-ranked spillover is a feature.

## Recommendation (ranked, honest)

1. **Do not chase density as a QLIKE lever.** The evidence (Spearman +0.19 density-vs-benefit vs +0.73
   signal-vs-benefit; densest config = weakest benefit) says density is decoupled from performance.
   Report this as the primary finding.
2. **If any A/B is run, test only Top-K=10 on SP500 (all horizons), single variable.** It is the sole
   denser variant whose added edges are the *strongest-ranked* survivors of a screen that currently
   discards 313 of 318 significant sources per target at h1. Low cost, testable, and the added edges
   are signal, not noise. **Ceiling is low**: even where VolGA already beats LSTM on SP500, HAR-X beats
   VolGA (`benefit_vs_harx` = -0.14 to -0.16) -- so more edges will not close the deep-vs-linear gap;
   at best it widens the (already significant) VolGA-vs-LSTM margin. Frame it as a robustness check on
   the Top-K choice, not a path to beating HAR-X.
3. **Reject FDR/BH and relaxed-alpha as denser edges.** They add thousands of tiny-|r| (max<0.25,
   median~0.05) edges -- statistically "significant" only because N is large -- reproducing the
   2026-08-29 over-full failure mode. Do not train these.
4. **Do not add vol->vol or |return|->vol edges** for a QLIKE gain: sources are plentiful (Top-K
   capped) but encode comovement the LSTM already sees, and the 2026-08-29 probes found no lift.
5. **The real gain ceiling is the intrinsic weak cross-sectional spillover on daily volatility**
   (max |r| < 0.25, decaying to noise by h22), not the edge-construction knobs. The significance
   screen is correctly keeping the signal and correctly falling back to no-graph where there is none.
   Effort is better spent elsewhere (e.g. the deep-vs-HAR-X gap on SP500) than on densifying the graph.

## Reproduce

```
.venv_gpu_encode/Scripts/python.exe scripts/eda/volga_edge_sparsity_eda.py     # -> *.json
.venv_gpu_encode/Scripts/python.exe scripts/eda/volga_edge_sparsity_plots.py   # -> *.html
```
Params: lookback=10, folds_target=7, alpha=0.05, Top-K=5, min_pairs=30 (canonical, matching the
delivered runs' `config` block).
