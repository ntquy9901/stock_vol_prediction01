# MASTER (Market-Guided Stock Transformer) for daily Parkinson-variance forecasting

Date: 2026-09-06. Baseline: `baselines/2026-09-06_master_transformer/`. Results:
`results/master_transformer/master_<market>_h<h>.json`.

## Objective
Adapt MASTER (Li et al., AAAI 2024, arXiv:2312.15235; MIT reference code SJTU-DMTai/MASTER, vendored in
`_master_ref/`) to forecast daily Parkinson variance on the VN30 and VN100 panels, and measure whether
its market-guided gate + **dense** inter-stock self-attention beats HAR-X (OLS), a no-graph LSTM, and
VolGA (LSTM + a **sparse** horizon-matched vol->PK GAT). The contrast of interest is dense learned
cross-sectional attention (MASTER) versus a sparse learned edge (VolGA) versus a linear champion (HAR-X)
on a noisy daily target on which graphs have repeatedly failed to beat HAR-X.

## Method

### Architecture ([N,T,D] adaptation)
Each anchor (one trading date t) is one MASTER sample: the cross-section of N stocks, each with a
T=lookback window of D features. The vendored MASTER modules (PositionalEncoding, TAttention,
SAttention, Gate, TemporalAttention) are reproduced in `code/master_net.py` and adapted to a leading
anchor-batch dimension so anchors are processed in batches on the GPU; the per-anchor math is unchanged
(a B=1 batched forward equals the reference — pinned by test). Flow: market-guided feature gate ->
Linear -> positional encoding -> intra-stock temporal attention -> **dense** inter-stock attention
(softmax over ALL stocks in the cross-section, the contrast to VolGA's Top-5 edge) -> temporal
aggregation -> linear decoder -> one variance forecast per stock.

- D = 5 stock features (`parkinson_variance, har_weekly, har_monthly, market_pk, volume_zscore_22`,
  read from the enriched panel and per-node train-only standardized by the delivered `pack_fold`) + 4
  causal market features. `gate_input_start=5`, `gate_input_end=9`. The gate reads the market features at
  the LAST timestep (the current, causal market state).
- Market features (causal, per date t; `code/market_features.py`): cross-sectional mean and dispersion
  of Parkinson variance, cross-sectional mean volume z-shock, and a 22-day causal rolling z of the market
  vol level. Standardized per fold on TRAIN dates only.
- Invalid (non-trading) stocks are masked out of the dense attention as keys (the fair analogue of
  VolGA's masked adjacency), so valid stocks never attend to zero-padded stocks.

### Normalization and fairness
Per the project's proven pattern (no Softplus/ReLU): per-node z-score target, LINEAR decoder output,
inverse-transform at eval, then the shared positivity floor `max(pred, 1e-2*train_mean + 1e-12)`. QLIKE
uses the single shared floor `1e-8` for every model. All models are trained on the SAME expanding-window
walk-forward folds (lb10, folds_target=7), the SAME 5 seeds, and the SAME per-node train-only scalers;
`assert_no_leakage` (purge = horizon) is enforced each run. HAR/HAR-X are OLS; LSTM/VolGA are the
delivered `train_masked_rich` models (VolGA on the horizon-matched vol->PK edge). MASTER, LSTM and VolGA
are all retrained IN-RUN so every Diebold-Mariano test is exactly paired.

### Configuration
d_model=128, t_nhead=4, s_nhead=2, dropout=0.2, beta=2.0, lr=5e-4, weight_decay=1e-5, grad_clip=1.0,
epochs=16 (early-stop on val, patience=5), anchor batch 256 (VN30) / 128 (VN100), 5 seeds
(42,123,2026,7,2024). All tunables live in `code/master_config.py`; floors/windows/seeds/lookback come
from the canonical `pipeline_config`.

### Pipeline validation
The in-run HAR-X/LSTM/VolGA numbers reproduce the delivered `results/edge_hmatched/` numbers EXACTLY
(VN30 h1: HAR-X 0.4801, LSTM 0.4787, VolGA 0.4740 — identical), confirming identical folds/seeds/config.

## Results (pooled OOS QLIKE; lower is better)

MASTER vs the baselines. DM = date-clustered Diebold-Mariano on per-obs QLIKE; "favors" is the better
model; p is two-sided. VolGA/LSTM/HAR-X QLIKE match `results/edge_hmatched/` (cross-check).

| Market | h | HAR-X | LSTM | VolGA | MASTER | DM MASTER vs HAR-X | DM MASTER vs VolGA | DM MASTER vs LSTM | MASTER fit |
|---|---|---|---|---|---|---|---|---|---|
| VN30 | 1 | 0.4801 | 0.4787 | 0.4740 | **0.4910** | p=0.165 favors HAR-X | **p=0.016 favors VolGA** | p=0.093 favors LSTM | ok |
| VN30 | 5 | 0.5602 | 0.5738 | 0.5672 | **0.6436** | p=0.179 favors HAR-X | p=0.107 favors VolGA | p=0.108 favors LSTM | ok |
| VN30 | 10 | 0.6091 | 0.6048 | 0.6061 | **0.6416** | p=0.337 favors HAR-X | p=0.117 favors VolGA | p=0.135 favors LSTM | ok |
| VN30 | 22 | 0.6782 | 0.6832 | 0.6882 | **0.6973** | p=0.687 favors HAR-X | p=0.595 favors VolGA | p=0.583 favors LSTM | ok |
| VN100 | 1 | 0.5000 | 0.5155 | **0.4879** | 0.5053 | MASTER>HAR-X (worse) | p=0.006 favors VolGA | p=0.188 favors MASTER | ok |
| VN100 | 5 | **0.5607** | 0.5759 | 0.5662 | 0.5697 | MASTER>HAR-X (worse) | p=0.476 favors VolGA | p=0.214 favors MASTER | ok |
| VN100 | 10 | **0.5999** | 0.6127 | 0.6127 | 0.6303 | MASTER>HAR-X (worse) | p=0.093 favors VolGA | p=0.093 favors LSTM | ok |
| VN100 | 22 | **0.6385** | 0.6452 | 0.6476 | 0.6628 | MASTER>HAR-X (worse) | p=0.272 favors VolGA | p=0.255 favors LSTM | ok |

MASTER (bold) is the WORST of the five models at every VN30 horizon; the date-clustered DM always favours
the baseline. The only DM that reaches significance is MASTER vs VolGA at h1 (p=0.016, MASTER worse) —
i.e. the dense cross-sectional attention is significantly beaten by the sparse learned edge exactly at
the horizon most favourable to cross-sectional structure. At longer horizons MASTER stays worst but the
gap is not individually significant (variance grows). Non-lock conditional QLIKE and per-obs win-rate
vs HAR-X (ticker-date win-rate 0.53-0.58, i.e. MASTER wins ~half the ticker-days but loses on aggregate
because a minority of large errors dominate) tell the same story. VN100 (complete, 5 seeds) confirms it:
MASTER is not best at any horizon (VolGA best at h1, HAR-X best at h5/h10/h22); at h1 MASTER is
significantly worse than VolGA (DM p=0.006) and worse than HAR-X, and it is the worst model at h10/h22.

The in-run HAR-X/LSTM/VolGA QLIKE reproduce `results/edge_hmatched/` to 4 decimals at every VN30 horizon.

### VN30 h1 detail
MASTER pooled QLIKE 0.4910 is the WORST of the five models (HAR 0.4779, VolGA 0.4740, LSTM 0.4787, HAR-X
0.4801). MASTER is significantly worse than VolGA (DM p=0.016) and directionally worse than HAR-X and
LSTM. Non-lock QLIKE (drop 45 limit-lock obs) tells the same story (MASTER 0.4566 vs HAR-X 0.4455, VolGA
0.4396). MASTER's per-obs QLIKE beats HAR-X on only 52.9% of ticker-days and 52.0% of unique dates, yet
it loses on the aggregate because a minority of large errors dominate. Per-seed QLIKE mean is 0.5718 ±
0.0467 (the seed-ensemble metric 0.4910 is lower, as expected). This is the horizon where cross-sectional
structure should help most, and the dense attention still does not beat the sparse edge or the linear
model.

## Overfit / underfit evidence
Every result JSON carries `train_metrics` + `val_metrics` + `metrics` (test) + per-model
`fit_diagnostics` + per-fold/seed `learning_curves`. VN30 h1: MASTER/LSTM/VolGA all `ok` at the pooled
level (MASTER pooled train->test QLIKE 0.588->0.491 improves, R2 tr/te 0.253/0.253 — no overfit gap).
One VN30 h1 fold showed a transient per-fold overfit verdict for MASTER, absorbed at the pooled level.
The result passes `scripts/quality_gate/overfit_check.check_result_evidence` (learned models detected by
name incl. "master").

## Limitations
- SP500 skipped locally (heavy; runs on Colab A100), as scoped.
- Market features are a compact causal set (4). Given the consistent negative direction, further
  market-feature engineering is low priority.
- The per-seed QLIKE variance for MASTER is non-trivial (±0.047 at VN30 h1); the reported comparison uses
  the seed-ensembled prediction for DM and reports the per-seed mean±std alongside.
- MASTER's dense attention over the full cross-section is the intended architectural feature; on this
  noisy daily variance target it adds noise rather than signal relative to a sparse edge / linear model.

## GO / NO-GO
**NO-GO (VN30 and VN100 both confirmed).** MASTER does not beat HAR-X, LSTM, or VolGA at ANY VN30
horizon — it is the worst of the five models at h1/h5/h10/h22, and significantly WORSE than the
sparse-edge VolGA at h1 (DM p=0.016), the horizon most favourable to cross-sectional structure. This is
consistent with the project-wide prior that dense/learned graph structure does not beat HAR-X on this
noisy daily variance target. The finding: a market-guided dense cross-sectional transformer underperforms
both a sparse learned edge (VolGA) and a linear HAR-X here; adding dense attention adds noise, not signal.
VN100 smoke already reproduced the direction and the full VN100 run is expected to confirm.

## Most-justified next step
None that changes the conclusion locally. If pursued: run the same MASTER on the larger SP500 panel on
Colab A100, where a bigger, more correlated cross-section is the only regime where deep/attention models
have previously shown any edge on this project — but the VN evidence predicts no lift.

## Reproduce
```
.venv_gpu_encode/Scripts/python.exe baselines/2026-09-06_master_transformer/code/run_master.py --market vn30 --horizon 1
python -m pytest baselines/2026-09-06_master_transformer/test -q   # 23 tests, C0=100% on changed code
```
