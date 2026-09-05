# Horizon-matched vol->PK edge — design

## Data flow
`build_enriched_panel` -> per fold: `pack_fold` (delivered fixed lag-1 edge `D.adj_vol2pk`) +
`directed_vol2pk_hmatched` (new horizon-matched edge) -> `RMR.train_masked_rich` trains the no-graph
LSTM (identity adj), VolGA (fixed edge) and VolGA_hm (horizon-matched edge) per seed -> pooled
predictions -> `RMR._metrics` + `RMR._dm_all` -> JSON. HAR/HAR-X from `_har_ols_preds`.

## Key decision — the edge builder
`directed_vol2pk_hmatched(vshock, sqrt_pk, last_row, horizon, top_k, alpha)`:
- horizon-matched lead-lag: `src = volume[:-h]`, `tgt = sqrt_pk[h:]` (vs the delivered `[:-1]`/`[1:]`).
- Bonferroni significance floor: keep source `i` for target `j` only if
  `|corr_ij| > z_bonf / sqrt(n_pairs_ij)`, `z_bonf = Phi^-1(1 - alpha/(2(n-1)))` (stdlib
  `statistics.NormalDist`). A per-pair `z/sqrt(m)` floor is too weak (large `m`; Top-K over ~100
  sources always clears it) — Bonferroni corrects for the multiple sources tested per target.
- Top-K by `|corr|` among survivors; self-loop = 1; empty row -> self-loop only (no-graph fallback).
- Train-only: uses rows `<= last_row = last train anchor + horizon` (mirrors the delivered edge).

## Anti-abstraction / isolation
Reuses `build_enriched_panel`, `pack_fold`, `train_masked_rich`, `_har_ols_preds`, `make_folds`,
`_metrics`/`_dm_all` read-only. Only new code = the edge builder + density diagnostic. No delivered
file is modified. `EDGE_SIG_ALPHA` is the one new tunable (documented at module top).

## Diagnostic
Per-fold edge density (fraction of non-self entries kept) is recorded for both the fixed and
horizon-matched edges, to show the fallback: dense where a real lead-lag exists (short h), sparse at
long h where the screened signal is noise.

## Performance
Batched masked-panel GPU training (delivered `train_masked_rich`); one adjacency built per fold. Three
model variants per seed (LSTM, VolGA, VolGA_hm) trained sequentially; runs per-horizon to avoid 8 GB
VRAM thrash.
