# Morning brief — multimarket paper (overnight 2026-09-14)

Target paper: `docs/paper/soict_2026-09-12_multimarket.tex` (+ .pdf). Submission: today noon.

## DONE overnight (committed + pushed, gate green)

1. **HOSE table -> OWN-8** (commit `4a4b1160`). `tab:hose` re-rendered from the committed `full_compare_hose.json`
   (rq dropped). Earnings on HOSE is now a **clean null at every horizon** (DM GBME vs GBM p=0.79/0.37/0.79/0.88),
   replacing the with-rq "p=0.000 at h1". Oracle sentence updated (helps h1/h5/h10, n.s. at h22). HARQ source
   `paper_metrics_hose.json` committed. PDF rebuilt (11 pp).
2. **Graph-null robustness sentence** added to HOSE: the two later ablations (concat 7-feature spillover, residual
   refiner) also fail DM -> graph carries no origin-time signal at feature or residual level.
3. **HOSE spike-robustness** (commit `e0a38b11`, new `scripts/eda/hose_spike_robustness.py` + tests + JSON).
   GBM beats HAR in **every fold**; the margin **survives excluding COVID/2022/Apr-2025** shock windows (57,466
   rows, 14%): h1 gain 13.3% -> 13.8% (DM p<1e-25). Paragraph added to the paper.
4. **GNNHAR SP500 baseline + resilient Colab notebook** (commit `d4ff03c2`, verified 2-layer: 11 tests, fresh-clone
   import smoke clean, 3-layer review, gate green). Run link:
   `https://colab.research.google.com/github/ntquy9901/stock_vol_prediction01/blob/master/notebooks/gnnhar_sp500_colab.ipynb`

## NEEDS YOU / decisions (prioritized)

1. **[MUST] Finish SP500 OWN-8 + align the Method.** The SP500 Colab (`sp500_final_models_colab.ipynb`) only
   delivered **h1** before the session dropped (laptop sleep). Re-open it to finish h5/h10/h22. **Current paper
   is mixed:** HOSE table = OWN-8 (R^8, no rq); Method text + SP500 table still describe **R^9 with rq**. This
   is an internal inconsistency to resolve. SP500 OWN-8 h1 = GBM 0.3572 / GBME 0.3173 / +graph 0.3159 is within
   0.0002 of the current 9-feature numbers, so the conclusions do not change. Once the SP500 Colab finishes all
   4 horizons I will (a) swap `tab:sp500` to OWN-8 and (b) change the Method own-history vector from R^9 to R^8
   (remove the realized-quarticity `rq` feature + its equation). ~15-min mechanical step. Decision: proceed with
   full OWN-8 (recommended, matches your HOSE decision) or keep rq everywhere (would require reverting HOSE).
2. **[OPTIONAL] GNNHAR row.** GNNHAR HOSE h1 confirms the thesis: GNNHAR 1.6823 beats HAR (p<1e-137) but **loses
   to GBM** 1.5702 (p<1e-83), and its **graph component hurts** (no-graph 1.5800 < graph 1.6823, p<1e-203).
   Full 4-horizon HOSE training was still running; the SP500 GNNHAR run is yours to start (link above). If you
   want GNNHAR in the paper, it is a strong extra data point against the graph literature; I can add a row/sentence
   once 4 horizons are in.
3. **[OPTIONAL] GARCH row.** `garch_hose.json` ready (GBM < GARCH < HAR at h1/h5/h10). Adding it needs a footnote
   (~1.4% short-history rows excluded) and a caveat (GARCH does not beat HAR at h22, p=0.589). Say the word.

## State of the numbers (all committed)
- HOSE: `full_compare_hose.json` (OWN-8, 8 folds x 3 seeds x 4 h), `garch_hose.json`, `hose_spike_robustness.json`,
  `gbm_earn_graph_hose.json` + `residual_graph_hose.json` (graph ablations, both NO-GO).
- SP500: `full_compare_sp500.json` (OWN-8, **h1 only** so far), `garch_sp500.json`.
- GNNHAR HOSE: worktree `agent-a6cf400f89676ba02` (h1 done + full overfit evidence; h5-22 pending).

## Conclusions unchanged by the OWN-8 switch
Own-history gamma-GBM beats HAR/HARQ (and GARCH) on both markets; cross-firm graphs add no origin-time QLIKE
value (three independent designs fail DM); forward-looking earnings is the S&P 500 lever and does not transfer
to HOSE. The OWN-8 switch only makes the HOSE earnings result a cleaner null and drops one redundant feature.
