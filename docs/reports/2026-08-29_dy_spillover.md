# DY (2014) volatility-spillover graph edge — HNX ablation

Diebold & Yilmaz (2014) directed variance-decomposition connectedness tested as the GAT graph edge for
the HNX volatility model, against the no-graph LSTM and the shipped statistical directed vol->PK edge
(and the sector-GAT for context). Only the EDGE differs; the LSTM branch, 5 node features, masked panel,
HAR-X anchor, per-ticker scalers and QLIKE evaluation are identical (reuses the delivered
`MaskedRichNet` / `train_masked_rich`).

## 1. DY (2014) equation mapping + VAR/LASSO choice

Connectedness = row-normalised GENERALIZED forecast-error-variance decomposition (generalized FEVD of
Pesaran-Shin 1998, adopted by DY 2014 so the network is invariant to variable ordering), estimated on the
TRAIN Parkinson-variance panel only and frozen for validation/test.

| DY (2014) object | Implementation (`code/dy_connectedness.py`) |
|---|---|
| `x_t = sum_{i=1}^p Phi_i x_{t-i} + eps_t`, `Sigma=cov(eps)` | `fit_var_elasticnet` — elastic-net VAR per equation |
| `A_0=I`, `A_h = sum_{i=1}^p Phi_i A_{h-i}` (VMA(inf)) | `vma_from_var` |
| `theta_ij(H) = sigma_jj^{-1} sum_{h<H}(e_i' A_h Sigma e_j)^2 / sum_{h<H}(e_i' A_h Sigma A_h' e_i)` | `generalized_fevd` |
| `theta_tilde_ij = theta_ij / sum_k theta_ik` (rows sum to 1) | `generalized_fevd` (row-normalise) |
| directed edge j->i = `theta_tilde_ij` = `A[target i, source j]` | `to_adjacency` (Top-K sources + self-loop=1.0) |

An INDEPENDENT loop-based recompute of the generalized-FEVD equation is asserted equal to the vectorised
implementation (`test_gfevd_matches_independent_reference_formula`), and the VMA recursion is verified
independently (`test_vma_recursion_two_lags`).

**High-dimensionality (HNX N=154):** a full unregularised VAR is ill-posed, so the standard high-dim
connectedness fix of Demirer, Diebold, Liu & Yilmaz (2018, *JAE* 33(1):1-15) is used — each VAR equation
is estimated with an elastic-net penalty. **Choice:** VAR(1), elastic-net `alpha=0.05`, `l1_ratio=0.5`,
per-ticker z-scored train series (so a single penalty is meaningful across the tiny-magnitude variance
series), FEVD horizon `H=10` (DY's standard). Estimated on rows strictly before the first validation
target date (`D.d_va[0]`); frozen. The model adjacency keeps the Top-K=5 spillover sources per node
(matching the vol2pk sparsity for a fair edge-only comparison) with a unit self-loop.

## 2. Connectedness-matrix statistics (full theta_tilde, HNX N=154, train)

Computed on the full screened HNX universe (N=154, n_train_rows=4124), VAR(1) elastic-net (alpha=0.05,
l1_ratio=0.5), FEVD H=10 (`results/dy_spillover_ablation/ckpt/dy_stats.json`):

- Total connectedness index (DY `C`): **44.7%** (44.73% of total forecast-error variance is cross-firm
  spillover; the rest is own-variance) — a plausibly-high, non-degenerate spillover network.
- Row-sum mean / min / max (normalisation check, target 1.0): **1.0000 / 1.0000 / 1.0000** (exact).
- Mean directional FROM-others = TO-others: **0.447** (network average balances by construction).
- Max TO-others (biggest transmitter, out of N): **1.329** (one HNX name transmits ~1.3 firms' worth of
  variance — a hub).
- Mean own-variance share (diagonal): **0.553**.
- Off-diagonal asymmetry (Frobenius ||theta - theta'||): **0.611** (> 0 => genuinely DIRECTED, not a
  symmetric correlation graph).
- Model adjacency = Top-K=5 spillover sources per node + self-loop (avg off-degree 5.0).

## 3. Metric table — HNX h1, 10 epochs, seeds {42,123,2026} (GPU)

N=154, n_test=60028 (390 test dates), all fit verdicts = **ok** (no over/under-fit). Numbers from
`results/dy_spillover_ablation/dy_ablation_hnx_h1.json`. QLIKE (ens) = metric of the seed-averaged
prediction (used for DM); QLIKE (seed) = mean +/- std of the per-seed QLIKE (the honest multi-seed figure).

| model | MSE | RMSE | MAE | QLIKE (ens) | QLIKE (seed, mean+/-std) | R2 |
|---|---|---|---|---|---|---|
| dy_GAT | 1.394e-6 | 0.001181 | 0.000645 | 1.9192 | **2.203 +/- 0.424** | 0.2201 |
| stat_GAT_vol2pk | 1.392e-6 | 0.001180 | 0.000653 | 1.8271 | 1.832 +/- 0.007 | 0.2213 |
| no_graph_LSTM | 1.390e-6 | 0.001179 | 0.000655 | 1.8301 | 1.835 +/- 0.008 | 0.2222 |
| sector_GAT (context, separate run) | 1.381e-6 | 0.001175 | 0.000639 | 1.8181 | - | 0.2276 |

Per-seed QLIKE: dy_GAT = [1.899, **2.803**, 1.908] (one seed collapses => high variance); stat =
[1.842, 1.828, 1.827]; no_graph = [1.847, 1.828, 1.832]. On MSE/RMSE/MAE/R2 all four edges are within
one seed's noise (RMSE differs at the 4th decimal).

## 4. Diebold-Mariano (date-clustered)

| comparison (A vs B) | QLIKE mean_diff | QLIKE p | favors | SE (MSE) p | AE p |
|---|---|---|---|---|---|
| dy_GAT vs no_graph_LSTM | +0.0985 | <0.001 | **no_graph** | 0.185 (ns) | <0.001 |
| dy_GAT vs stat_GAT_vol2pk | +0.1016 | <0.001 | **stat** | 0.376 (ns) | <0.001 |
| stat_GAT_vol2pk vs no_graph_LSTM | -0.0031 | 0.068 (ns) | stat | 0.268 (ns) | <0.001 |

(mean_diff < 0 favors A.) The QLIKE loss is significantly WORSE for dy_GAT than for both no-graph and the
statistical edge; on squared-error (MSE basis) the edges are statistically indistinguishable (SE p > 0.18
everywhere). sector-GAT: recorded QLIKE shown for context only; no per-obs DM (its predictions were not
stored by the separate sector run). The stat-vs-no-graph QLIKE null (p=0.068) reproduces the sibling
edge-ablation finding.

## 5. Verdict

**The DY (2014) spillover edge does NOT improve HNX h1 volatility forecasts — on QLIKE it significantly
underperforms** both the no-graph LSTM and the statistical vol->PK edge (DM p<0.001, both favoring the
non-DY model), while on squared-error / MSE the four edges are statistically indistinguishable
(DM-SE p>0.18). The dy_GAT QLIKE is also seed-unstable (one of three seeds collapses to 2.80 vs ~1.90),
whereas the stat and no-graph edges are tight (std ~0.007-0.008). Fit diagnostics are "ok" for all three
(no over/under-fit), so this is a genuine no-lift result, not a training pathology.

This is the **fourth edge to fail to beat the no-graph LSTM on HNX h1**, alongside the statistical vol->PK
edge (QLIKE null, p=0.068), the static sector edge, and the MTGNN learned edge. The consistent picture
across four structurally-different graphs — a data-driven directed variance-decomposition network
(this work), a lead-lag statistical edge, static sector metadata, and an end-to-end learned adjacency —
is that **the graph/spillover structure adds no out-of-sample value for HNX one-day volatility**; a
parsimonious no-graph LSTM (and HAR) is preferred. The correctly-constructed, genuinely-directed DY
connectedness network (total connectedness 44.7%, section 2) makes this a strong robustness statement:
even the "textbook" spillover network does not help.

## 6. Reproduce / commands

```
# unit + smoke tests (23 tests, C0=100% C1=100% on all three modules)
python -m pytest baselines/2026-08-29_dy_spillover_ablation/test/ -q

# dry (build DY adjacency + one forward pass, tiny slice)
DY_ABLATION_FORCE_CPU=1 python baselines/2026-08-29_dy_spillover_ablation/code/run_dy_ablation.py \
  --panel hnx --horizon 1 --max-tickers 12

# full run (HNX h1, 10 epochs, 3 seeds, GPU; checkpointed + resumable)
DY_ABLATION_FORCE_CPU=0 .venv_gpu_encode/Scripts/python \
  baselines/2026-08-29_dy_spillover_ablation/code/run_dy_incremental.py \
  --panel hnx --horizon 1 --epochs 10 --seeds 42 123 2026
```

## 7. Notes

- Ran on GPU (RTX 4060) once the machine was free (sector-GAT + MTGNN finished); ~25 min for 9 trainings
  (3 variants x 3 seeds x 10 epochs). Checkpointed per (variant, seed) so a kill loses at most one
  in-flight training.
- Interpreter: `.venv_gpu_encode` (`torch 2.6.0+cu124`, the venv the pre-push gate uses). System Python
  3.14 + `torch 2.12.1+cpu` segfaults the LSTM training loop — do not use it for training.
- Over/under-fit evidence: `result.json` carries `train_metrics` + `val_metrics` + `fit_diagnostics`
  (all "ok") + per-seed `learning_curves` for every variant, per the CLAUDE.md mandate.
- Performance: DY-matrix build is a one-off CPU/VAR step; training reuses the delivered batched
  `[B,N,...]` pipeline (batched block adjacency, mask-aware loss) — no batch=1 anti-pattern.
