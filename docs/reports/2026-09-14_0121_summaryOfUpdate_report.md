# Summary of Update — GBM residual-graph refiner (falsification baseline, HOSE)

Date: 2026-09-14 01:21

## What changed
New baseline `baselines/2026-09-14_gbm_residual_graph/` — a pre-registered falsification of the one graph-fusion
direction not previously tried: train a graph-spillover refiner on the GBM(own-8) RESIDUAL (the reference
`C:\research\gnn` / arXiv:2410.16858 idea), rather than concatenating graph features into the GBM (prior
attempt a) or a plain GNN (prior attempt b). HOSE result: `results/gamma_gbm/residual_graph_hose.json`.

## Method
Base = seed-averaged gamma-GBM(own-8), OOS per walk-forward fold (identical to full_compare). A ridge is
trained on EARLIER folds' OOS log-variance residuals (expanding, causal) from causal neighbour-spillover
features (top-10 train-only correlation graph), and reconstructs `final = gbm·exp(resid_hat)`. Two variants:
`raw` (unclipped) and `clip` (winsorised target + prediction clipped to ±0.5, i.e. adjustment in [0.61×,1.65×]).
GO verdict is judged on the fair `clip` variant.

## HOSE result (QLIKE, lower better; DM vs GBM, date-clustered)
| h | GBM | raw | clip (fair) | clip gain | clip DM p | verdict |
|---|---|---|---|---|---|---|
| 1 | 1.5702 | 5.047 | 1.5758 | −0.36% | 0.068 | NO-GO |
| 5 | 1.6484 | 5.335 | 1.6506 | −0.13% | 0.694 | NO-GO |
| 10 | 1.6840 | 5.627 | 1.6848 | −0.05% | 0.912 | NO-GO |
| 22 | 1.7263 | 6.228 | 1.7226 | +0.21% | 0.702 | NO-GO |

`success=False`. The bounded (fair) refiner is statistically indistinguishable from GBM at every horizon
(DM p>0.05 everywhere; the h22 +0.21% is not significant). The graph carries no orthogonal OOS signal on top
of own-history — the predicted mechanism. The unbounded `raw` variant is QLIKE-catastrophic (−221%…−261%): the
log-MSE-fit / variance-QLIKE mismatch amplified by HOSE floor/limit-lock residuals through `exp(.)` — the exact
scale-sensitivity the reference paper warns about. This is the third structurally-distinct graph attempt to
fail the QLIKE+DM arbiter, strengthening the thesis' negative graph result.

## Tests / coverage
8 tests (`.venv_gpu_encode` pytest), all pass: reconstruction positivity (both variants), causal expanding
stack (prefix invariance — no future-fold leakage), linear-signal recovery, noise neutrality, verdict logic,
run structure + per-horizon checkpoint, empty-when-no-fold. Diff-coverage evaluated at push (C0 100% / C1 ≥95%
on changed lines; `main()` `# pragma: no cover`).

## Code review
3-layer adversarial (`code_review/code_review_2026-09-14.md`): no critical/major. Two minor, conclusion-safe
caveats documented — (1) the expanding stack uses adjacent folds without an extra label-embargo (a ~h-day
boundary bleed that can only INFLATE the refiner's power, yet it is still neutral → NO-GO robust); (2) a <2-fold
market could hit a DM zero-variance error (not reachable on HOSE/SP500's 8 folds). Performance lens: GBM batched
seed-avg + vectorised ridge; no batch=1 anti-pattern.

## Data-quality gate
N/A — no data change (reads existing enriched HOSE frames only).

## SP500
Not run (falsification is conclusive on HOSE and the mechanism is market-agnostic). Can be added to a Colab
notebook later if a two-market table is wanted; the code takes `sp500` as-is.

## DoD
Code ✓ · TDD 8 tests ✓ · code review ✓ (no critical/major) · gate (at push) · summary ✓ · result committed.
Follow-up: none required — pre-registered NO-GO reached; graph remains no-go on the arbiter.
