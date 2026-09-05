# Horizon-matched vol->PK edge — requirements

**Type:** edge-construction fix / ablation (reuses the delivered VolGA model; changes only the graph).

## Problem
The delivered vol->PK edge (`masked_rich._directed_vol2pk`) always correlates `volume_i(t)` with
`sqrt_pk_j(t+1)` — a FIXED 1-day lead-lag — regardless of the forecast horizon `h`. It matches the
target at h1 but is mis-aligned at h5/h10/h22 (model forecasts vol at `t+h`, graph encodes `t->t+1`),
a plausible cause of the graph's instability at longer horizons.

## Goal
Test a horizon-matched edge `corr(volume_i(t), sqrt_pk_j(t+h))` plus a Bonferroni significance floor
(auto-fallback to no-graph where the lead-lag is at noise level), holding the model, features, folds,
seeds and DM protocol fixed.

## Input / output
- Input: enriched walk-forward panels (`enriched_glob(market)`), 5 node features, lookback 22.
- Output: `results/edge_hmatched/edgehm_{market}_h{H}.json` with `metrics` (HAR, HAR-X, LSTM, VolGA,
  VolGA_hm) + `dm_date_clustered` (VolGAhm-vs-VolGA, VolGAhm-vs-LSTM, VolGA-vs-LSTM) + per-fold mean
  edge density for the fixed and horizon-matched edges.

## Acceptance criteria
- Edge is train-only, horizon-matched (`t -> t+h`), self-loop=1, NaN-safe.
- At h=1 with the floor disabled the builder reproduces the delivered edge exactly (tested).
- The Bonferroni floor prunes spurious edges on noise vs the unfloored Top-K (tested).
- Every reported number traces to a stored results JSON; DM is date-clustered with HLN.

## Go / no-go
Reported honestly. VolGA_hm is called an improvement over the delivered edge only if DM
(VolGAhm-vs-VolGA) is significant AND the graph's marginal value over the no-graph LSTM does not
worsen. Expectation (per prior EDA): horizon-matching removes the mis-alignment artifact but the
long-horizon lead-lag signal is near noise, so VolGA_hm is expected to approach the no-graph LSTM at
h10/h22 rather than beat it.
