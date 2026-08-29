# Sector GAT — code and methodology review

**Date:** 2026-08-29  
**Scope:** `baselines/2026-08-29_sector_gat_ablation/` and integration with `MaskedRichNet`.  
**Mode:** read-only review; no source, data, or result files were modified.

## Executive verdict

The implementation is a valid first ablation: it builds ticker-aligned same-sector adjacency,
reuses the existing LSTM+GAT training path, handles unmapped tickers conservatively, and produces a
three-way comparison (`sector_GAT`, `stat_GAT_vol2pk`, `no_graph_LSTM`). The basic graph construction
and masking are coherent.

It is **not yet paper-ready as evidence that sector information is leakage-free or that GAT generally
helps**. Two issues must be resolved before a formal experiment:

1. The sector graph is a fully connected sector clique by default, while the statistical graph is
   Top-K. This changes graph density and message-passing capacity, so this is not a controlled
   edge-source swap.
2. The VN labels are a current snapshot from `vnstock`, applied to the complete historical sample.
   The snapshot is fixed after download, but it is not automatically point-in-time valid. The claim
   “zero leakage” is therefore too strong.

The available HNX result is a **directional smoke/training result only**: CPU, five epochs, one seed,
H1. It must not be used as the final paper table.

## Findings

### S-01 — HIGH: sector and statistical graphs are not capacity-matched

**Evidence**

- `code/sector_adjacency.py:34-43,56-67`: `top_k=None` creates a fully connected graph inside each
  sector.
- `code/run_sector_ablation.py:121-124`: the sector graph uses the default `top_k=None`, while
  `D.adj_vol2pk` was built with the existing statistical Top-K configuration.
- The HNX result reports sector average off-degree **10.81**, maximum **24**. The delivered
  statistical graph is capped by the existing Top-K setting (5).

**Why it matters**

The sector model receives many more possible neighbor messages. A GAT can learn attention weights,
but the candidate-neighbor set, aggregation variance, and effective capacity are still different.
A better result cannot be attributed only to the sector prior; it may partly be a density effect.

**Required action**

Run a controlled comparison with sector `top_k=5`, matching `MR.EDGE_TOP_K`, and preferably a
density sensitivity such as K=3/5/10. Report edge count, mean degree, degree quantiles, and largest
sector. If the clique is intentional, call the experiment “dense sector graph versus sparse
statistical graph”, not a pure graph-source ablation.

### S-02 — HIGH: “zero leakage” is not established by the current sector CSV

**Evidence**

- `code/fetch_vn_sectors.py:1-16,30-44`: one current `vnstock Listing().symbols_by_industries()`
  snapshot is used; `fetched_date` is only a supplied provenance string.
- `code/sector_adjacency.py:71-84`: the loader ignores effective dates and keeps the last duplicate
  row for a ticker.
- `code/run_sector_ablation.py:118-123`: one static map is applied to train, validation, and test
  history.
- `design/design.md:8-18` and `run_sector_ablation.py:8-10` claim zero leakage without a historical
  membership/effective-date check.

**Why it matters**

A current classification can reflect later reclassification, merger, listing, or survivorship
information. This is not a neural-forward-pass leak; it is a point-in-time metadata problem.

**Required action / acceptable treatment**

Use historical effective-date labels; or freeze labels at the training cut-off and show their
availability at that date; or explicitly call this an exogenous static metadata ablation, remove
“zero leakage”, and state the snapshot date/source as a limitation. The third option is acceptable
for exploration, not for a strong leakage-free claim.

### S-03 — HIGH: HNX result is underpowered, although the code labels it directionally

`results/sector_gat_ablation/sector_ablation_hnx_h1.json` contains CPU, 5 epochs, seed `[42]`, H1,
154 nodes and 60,028 valid test cells. It reports sector QLIKE 1.8921 versus 1.9164 (statistical
GAT) and 1.9153 (no-graph LSTM); DM QLIKE p-values are 0.0069 and 0.0101.

This is encouraging, but one seed and five epochs cannot establish stable paper performance. A large
cell count can also make p-values appear precise while dates and tickers remain correlated. The code
correctly labels this as a “quick DIRECTIONAL check, not a final number” at `run_sector_ablation.py:114-117`.

**Action:** after S-01/S-02, run the paper configuration with the same seeds, epochs, floors, data
universe, all horizons, and planned estimators. Keep this JSON as a directional artifact.

### S-04 — MEDIUM: this is same-sector message passing, not hierarchical sector GAT

`sector_adjacency.py:36-47` creates only ticker-to-ticker same-label edges. No sector nodes, sector
embeddings, industry-to-sector hierarchy, or explicit sector aggregation is added. The graph branch
uses the existing two-layer `WeightedGATLayer` stack and graph inputs from the last time step
(`run_masked_rich.py:104-116`).

The accurate description is **LSTM plus a static same-industry/same-sector graph message-passing
branch**. With a fully connected clique, the second layer does not expand reach beyond the same
sector; it mainly performs another within-block transformation. Do not claim multi-level sector
signals unless that hierarchy is implemented.

### S-05 — MEDIUM: sector mapping validation is not fail-closed

`fetch_vn_sectors.py:38-44`, `fetch_sectors.py:36-49`, and `sector_adjacency.py:77-84` silently use
last-write-wins for duplicate tickers. Conflicting classifications can therefore change the graph
without an error. `load_sector_map` also does not normalize case or dot/dash variants; normalization
exists only in fetch helpers, so a custom CSV can silently create singleton nodes.

**Action:** reject conflicting duplicates, validate ticker format/normalization, and record or fail on
unmapped tickers. Coverage percentage alone is not enough to detect this defect.

### S-06 — MEDIUM: result provenance is insufficient for reproducibility

`run_sector_ablation.py:135-143` records metrics and basic coverage, but not sector CSV hash, source
revision, sector level, `top_k`, adjacency nonzero count/density, degree quantiles, unmapped ticker
list, split date ranges, per-split valid-cell counts, config hash, or git commit. A mutable source URL
plus manually supplied `fetched_date` is not a complete immutable provenance record.

**Action:** record mapping SHA-256, source/version and level, snapshot/effective-date policy,
`top_k`, actual adjacency diagnostics, unmapped list, split ranges/counts, seeds/config, and commit.

### S-07 — LOW: coverage diagnostics can describe a different graph than the one used

`sector_adj_for(..., top_k=None)` can build a capped graph at `sector_adjacency.py:80-84`, but
`coverage()` always rebuilds a fully connected graph at `sector_adjacency.py:87-95`. The current run
uses `None`, so its numbers are consistent; a future capped run would report the wrong degree and
singleton statistics.

**Action:** pass `top_k` into `coverage()` or compute diagnostics from the actual adjacency matrix.

## What is correct

- Adjacency is ticker-aligned, float32 `[N,N]`, and has self-loops.
- Unmapped tickers become separate singleton nodes, avoiding invented cross-stock edges.
- The smoke path applies the same valid-source-node masking as the existing batched graph path.
- All three variants reuse the same model, target masks, metrics, seed plumbing, and date-clustered
  DM helper.
- The HNX result has equal valid test-cell count across models and high mapping coverage (153/154,
  99.35%), so it is internally comparable as a directional run.

## Recommended fix order

1. Resolve the point-in-time/snapshot policy and remove the absolute “zero leakage” wording unless it
   can be proven.
2. Run density-matched sector `top_k=5`; retain the clique only as a sensitivity analysis.
3. Add fail-closed mapping validation and diagnostics from the actual adjacency.
4. Expand JSON provenance and split diagnostics.
5. Re-run all paper horizons/panels/estimators with the full seed and epoch configuration.

## Final verdict

**Code:** structurally sound ablation prototype; no obvious tensor-shape or self-loop defect found in
the sector-specific code.  
**Publication readiness:** conditional fail until S-01 and S-02 are resolved.  
**Current HNX numbers:** promising directional evidence, not yet a defensible claim that sector GAT
is generally best or that the gain is caused by sector information alone.
