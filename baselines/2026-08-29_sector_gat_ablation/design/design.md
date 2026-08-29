# Sector-graph ablation — design

## 1. Rationale for the sector edge

The delivered graph mixes two statistical edges — Top-K correlation (`adj_corr`) and directed
volume-shock→Parkinson (`adj_vol2pk`). Both are *estimated on the train fold and frozen*, and the EDA
measured only ~9–30% edge persistence into the test fold: at test time the model attends over a graph
that barely overlaps the one it learned on. A GICS/ICB **sector** edge is different in kind:

- **Static metadata** — a company's sector does not drift between train and test, so there is zero
  edge-instability and zero look-ahead in the edge itself.
- **Economically motivated** — same-sector names share demand/cost shocks and co-move in volatility;
  the edge encodes a prior the statistical estimator has to (noisily) rediscover each fold.
- **Complements `market_pk`** — the market factor is already a node feature, so an all-to-all
  statistical graph is partly redundant with it; a *sector-block* graph adds strictly intra-sector
  structure the market node cannot express.

The ablation answers: with a leakage-free, drift-free edge, does the GAT branch finally help, or does
the graph still add no OOS value (as in prior runs)?

## 2. How the model consumes a graph (verified, read-only)

`run_masked_rich.MaskedRichNet` is a 2-hop `WeightedGATLayer` stack; `train_masked_rich(D, cfg, seed,
use_graph, adj, ...)` takes the `[N,N]` adjacency as a **parameter**. Training batches the adjacency
as `base[N,N] * valid_source_mask[b,1,N]` (`adj_batch`). So swapping the edge = passing a different
`adj` aligned to `D.tickers`; **no live file is edited**. `MaskedRichData` exposes `.tickers` (node
order), `.adj_vol2pk`, `.adj_corr` (self-loop, signed, float32).

## 3. Components (all new, in `code/`)

| File | Purpose |
|---|---|
| `sector_adjacency.py` | `build_sector_adjacency(tickers, sector_map, top_k)` → `[N,N]` float32; `load_sector_map`, `coverage`. Panel-agnostic. |
| `fetch_vn_sectors.py` | vnstock ICB → canonical `ticker,sector` CSV (VN). Lazy vnstock import (gate venv has no vnstock). |
| `fetch_sectors.py` | datahub GICS → canonical CSV (S&P500, retained). |
| `run_sector_ablation.py` | Build panel (reuse `_write_estimator_processed` + `build_masked_rich`), build `A_sector` aligned to `D.tickers`, dry forward-pass smoke, `--train-epochs N` CPU comparison. |

### Adjacency definition

`A[i,j] = 1` iff `i,j` share a sector; diagonal always 1 (self-loop). Unmapped tickers → **singleton
own-sector** (self-loop only — two different unmapped tickers are NOT merged into one "Unknown"
bucket). Default `top_k=None` = fully-connected-within-sector (symmetric/undirected). Optional `top_k`
caps off-diagonal degree per row by a stable criterion (node order); may be asymmetric. Weights are
**unit (0/1)**, matching the statistical adjacencies' unnormalized self-loop=1.0 convention so the same
`WeightedGATLayer` masking/attention applies unchanged (fair edge swap).

## 4. Design decisions & gates

- **Simplicity Gate:** one pure builder + one runner; reuse the shipped pipeline wholesale (no wrapper
  around the model, no new training code).
- **Anti-Abstraction Gate:** import `MaskedRichNet`/`train_masked_rich`/`build_masked_rich` directly;
  the sector edge is just a different `numpy` array.
- **Performance/Batching Gate:** training reuses the delivered batched GAT (`batch_size=512`,
  block-mask adjacency) — no per-item loop introduced. CPU is forced *only* to avoid contending with
  the live GPU job (env `CUDA_VISIBLE_DEVICES=""` set before `import torch`); the compute path itself
  is the delivered batched one. Set `SECTOR_ABLATION_FORCE_CPU=0` for GPU once free.
- **No silent degradation:** unmapped tickers are made explicit singletons and counted in `coverage`;
  a <2-file build raises; a non-finite forward output raises.

## 5. Data flow

```
raw OHLCV ─_write_estimator_processed(parkinson)→ processed CSVs ─build_masked_rich→ D (tickers, adj_vol2pk, X/y/masks)
vnstock ICB / GICS CSV ─load_sector_map→ sector_map ─build_sector_adjacency(D.tickers)→ A_sector[N,N]
D + {A_sector | D.adj_vol2pk | use_graph=False} ─train_masked_rich (CPU)→ preds ─_metrics/_dm_all→ result.json
```

## 6. Ready-to-run scale-up plan (after the GPU frees)

- **Panels:** HNX (primary), then VN100. **Target:** Parkinson. **Horizons:** 1 and 5.
- **Seeds:** the delivered 5 seeds `(42,123,2026,7,2024)`. **Epochs:** 5–10 (directional) → 20 for a
  final number if directional is promising.
- **Command (GPU, HNX, both horizons, 5 seeds):**
  ```bash
  for h in 1 5; do
    SECTOR_ABLATION_FORCE_CPU=0 .venv_gpu_encode/Scripts/python.exe \
      baselines/2026-08-29_sector_gat_ablation/code/run_sector_ablation.py \
      --panel hnx --horizon $h --train-epochs 10
  done
  # VN100: same, --panel vn100
  ```
- **Comparison table produced** (per panel × horizon), sector-GAT vs stat-GAT vs no-graph LSTM:

  | model | MSE | RMSE | MAE | QLIKE | R² |
  |---|---|---|---|---|---|
  | no_graph_LSTM | … | … | … | … | … |
  | stat_GAT_vol2pk | … | … | … | … | … |
  | sector_GAT | … | … | … | … | … |

  plus date-clustered Diebold–Mariano on QLIKE/SE/AE for **sector vs stat** and **sector vs no-graph**
  (favours the lower-loss model; `p<0.05` = significant).

## 7. Verification

TDD: failing property tests for `build_sector_adjacency` first, then implement. Real-data-sample CPU
smoke (tiny HNX slice) exercises the true writer + panel builder + a finite forward pass. Runner I/O
(`run_training`) tested with `train_masked_rich` monkeypatched to a deterministic stub (no GPU, no
epochs). Pre-push gate: C0 line 100% / C1 branch ≥95% on changed lines.
