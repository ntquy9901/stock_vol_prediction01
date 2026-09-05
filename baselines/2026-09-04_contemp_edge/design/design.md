# Contemporaneous-edge ablation — design

## Data flow
`build_enriched_panel` → per fold: `pack_fold` → `_fold_adj(edge)` picks the adjacency →
`RMR.train_masked_rich` trains no-graph LSTM (identity adj) and VolGA (contemp/vol2pk adj) per seed →
pooled predictions → `RMR._metrics` + `RMR._dm_all` → JSON.

## Key decision — the edge builder
`build_contemp_adj(sqrt_pk, last_row, top_k)`:
- Correlation `pandas.DataFrame(sqrt_pk[:last_row+1]).corr()` (pairwise-complete → NaN-safe), zeroed
  NaNs, cast float32.
- Per target row j, keep Top-K by `|corr|` off-diagonal sources; set self-loop = 1.
- `last_row = last train anchor + horizon` → strictly train-only, no look-ahead (tested).

`_fold_adj` dispatches: `edge="vol2pk"` returns the stored delivered adjacency `D.adj_vol2pk`;
`edge="contemp"` builds the contemporaneous edge from `panel.pk`.

## Anti-abstraction / simplicity
Reuses `run_masked_rich`, `run_walkforward`, `wf_enriched_panel`, `run_volga_walkforward` read-only.
Only new code = the two edge functions; everything else is the delivered pipeline. No new config
constants (Top-K, lookback, floors, seeds all come from the existing config modules).

## Performance
Batched masked-panel training on GPU (delivered `train_masked_rich`); one adjacency per fold built
once (cheap pandas corr on the train slice). Runs sequentially per horizon to avoid 8 GB VRAM thrash.

## Isolation
No files outside this baseline are modified. `run()` is a GPU training driver (marked no-cover);
unit tests cover the pure edge logic.
