# Requirements — GBM residual-graph refiner (falsification test)

## Motivation
The champion is a per-stock gamma-GBM on 8 own-history features. Two prior graph attempts failed on the
QLIKE + Diebold-Mariano (DM) arbiter: (a) feature-concat of 7 neighbour-aggregation columns INTO the GBM
(residual err_corr 0.96–0.98 vs own-history, h10 overfit blow-up), and (b) a plain 1-layer GNN predicting
volatility from scratch (diluted the own-AR signal, hurt every horizon). The reference repo `C:\research\gnn`
(arXiv:2410.16858) uses a structurally different fusion we have NOT tried: train the graph branch on the base
model's RESIDUAL, so the graph branch cannot re-explain own-history by construction (base model is the outer
level, graph is the refiner).

## Goal
Test one hypothesis, pre-registered as a falsification:

> H1: a causal graph-spillover refiner trained on the GBM(own-8) residual adds out-of-sample value on top of
> GBM(own-8), measured by pooled QLIKE (variance scale) + date-clustered DM.

## Input / Output
- Input: enriched per-stock frames for a market (`hose` or `sp500`) via `full_matrix.load`; the shared
  walk-forward folds (`S1.FOLDS`), top-10 per-fold correlation graph (`S1.build_graph`), causal neighbour
  features (`S1.graph_feats`).
- Output: `results/gamma_gbm/residual_graph_<market>.json` — per horizon: `n`, `qlike` {gbm, final},
  `gain_vs_gbm_pct`, `dm_final_vs_gbm` {p_value, mean_diff}, `fit_diagnostics`, `verdict`; plus a top-level
  `success` boolean from the kill criterion.

## Method (causal by construction)
For each horizon h and each walk-forward fold k (test window):
1. `gbm_pred(k)` = seed-averaged gamma-GBM on OWN-8, trained on train[<k] only (OOS, identical to full_compare).
2. Neighbour features `g_nb_*(k)` from the TRAIN-only top-10 correlation graph (day-t cross-section, no h-ahead
   target leakage).
3. Residual target `resid(k) = log(max(y(k),FL)) − log(max(gbm_pred(k),FL))` (log-variance residual).
4. Ridge refiner trained on the OOS residuals of EARLIER folds `{(g_nb_*(j), resid(j)) : j<k}` only (expanding,
   causal — never sees fold k's realized target), with train-only feature standardisation. Fold 0 → refiner = 0.
5. Reconstruct `final(k) = max(gbm_pred(k),FL) · exp(resid_hat(k))` (positive by construction).
6. Pool folds; QLIKE(final) vs QLIKE(gbm_pred); DM(final vs gbm_pred).

The refiner starts LINEAR (ridge) to minimise overfit surface (the h10 blow-up is the specific failure mode to
avoid). Escalation to a 1-layer GNN is out of scope unless the ridge shows a DM-positive gain.

## Success / Kill criterion (pre-registered)
- `success = True` only if, at BOTH h1 and h5: `gain_vs_gbm_pct > 0` AND `dm_final_vs_gbm.p_value < 0.05`.
- Otherwise `success = False` → NO-GO (graph adds no orthogonal OOS signal on the arbiter). A MAE-only or
  oracle-weighted "win" does NOT count.
- Expected outcome (pre-registered, per the reference's own QLIKE loss to HAR and our collinearity mechanism):
  neutral-to-negative on QLIKE+DM. A NO-GO strengthens the thesis; a GO is a genuine finding to escalate.

## Go / No-go for building
- GO to build: yes (agreed) — cheap (linear), reuses existing infra, and the residual direction is untried.
- Robustness: report all 4 horizons (1/5/10/22); run HOSE first, SP500 later via Colab. HOSE spike-robustness
  (per project rule) applies before any positive claim.

## Non-goals
- Not a champion attempt; not a GNN; no hyperparameter chase; no new tunable constants outside `config.py`.
