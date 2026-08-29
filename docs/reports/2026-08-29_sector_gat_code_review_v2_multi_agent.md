# Sector GAT — multi-agent verification review v2

**Date:** 2026-08-29  
**Scope:** sector-GAT ablation, graph construction, integration with `MaskedRichNet`, metadata,
evaluation, tests, and paper readiness.  
**Review method:** six independent agents reviewed separate aspects, then findings were consolidated.  
**Modification policy:** source code, data, and result files were not modified; only this report was added.

## Final verdict

### What is verified as correct

The sector adjacency implementation and tensor plumbing are internally correct:

- `[N,N]` float32 adjacency aligned to `D.tickers`;
- self-loop on every node;
- unmapped tickers become separate singleton nodes;
- cross-sector edges are absent;
- batched adjacency uses the expected `[B,N,N]` shape;
- invalid source nodes are masked consistently with the existing training path;
- target masks control scored observations;
- sector-GAT, statistical-GAT, and no-graph variants share the same prediction/metric/DM plumbing.

The targeted sector suite was reported by the agents as **34 tests passed**. Those tests establish
builder and wiring behavior, not full training validity or paper-level conclusions.

### What remains blocking for a paper claim

The experiment is currently a plausible exploratory prototype, but it does not yet prove that:

1. the sector comparison isolates sector information;
2. the labels are point-in-time leakage-free; or
3. the HNX result is stable across seeds, epochs, horizons, and panels.

Publication verdict: **CONDITIONAL FAIL** until S-01 and S-02 below are resolved or explicitly
reframed as limitations.

## Findings

### S-01 — HIGH: graph comparison is confounded by density, directionality, and weights

Evidence:

- `baselines/2026-08-29_sector_gat_ablation/code/sector_adjacency.py:34-46,56-67` uses a fully
  connected same-sector graph by default (`top_k=None`), with symmetric binary weights.
- `baselines/2026-08-29_sector_gat_ablation/code/run_sector_ablation.py:80-84,121-124` uses that
  default for `sector_GAT`, while `D.adj_vol2pk` uses the existing statistical Top-K graph.
- The statistical graph is directed and uses signed/magnitude correlation weights; sector edges are
  symmetric and all weight 1.0.
- The recorded HNX sector graph has average off-degree 10.81 and maximum 24. The statistical graph
  is capped at five outgoing non-self edges under the delivered configuration.

The sector model therefore changes at least three factors simultaneously: neighbor count, edge
direction, and edge-weight semantics. A sector improvement cannot be attributed only to sector
information.

**Required action:** run a density-matched sector graph with `top_k=5` (matching `MR.EDGE_TOP_K`),
and report a K sensitivity such as 3/5/10. Record edge count, density, degree quantiles, and largest
sector. If the clique is retained, describe the result as “dense sector graph versus sparse
statistical graph”, not as a pure edge-source ablation.

### S-02 — HIGH: point-in-time leakage is not established

Evidence:

- `code/fetch_vn_sectors.py:1-16,37-44` obtains one current `vnstock` snapshot.
- `vn_sectors.csv` and `sp500_gics_sectors.csv` contain `fetched_date`, but no
  `effective_from/effective_to` or historical classification version.
- `code/run_sector_ablation.py:118-123` loads one map and applies it to all historical train,
  validation, and test dates.
- `code/run_sector_ablation.py:8-10` and `design/design.md:8-18` call this “zero leakage”.

Static after download is not the same as point-in-time valid. Later reclassification, merger,
constituent, or survivorship information may be applied retroactively.

**Required action:** use historical effective-date labels; or freeze labels at the train cut-off and
prove availability then; or explicitly reframe this as an exogenous static metadata ablation,
remove “zero leakage”, and state snapshot date/source as a limitation.

### S-03 — HIGH: custom temporal split configuration is silently ignored

`code/run_sector_ablation.py:68-76` calls `MR.build_masked_rich()` with lookback and horizon but does
not forward `cfg.train_frac` or `cfg.val_frac`. The builder therefore uses its own defaults.

The current defaults may happen to coincide, but a custom split configuration can silently produce a
different experiment from the configuration recorded by the caller. This is a correctness issue for
reproduction and temporal robustness studies.

**Required action:** forward the configured split parameters, or remove them from the public config and
record the actual split values returned by the builder. Add a test that changes the config and asserts
the resulting date boundaries change accordingly.

### S-04 — HIGH: current HNX result is directional only

`results/sector_gat_ablation/sector_ablation_hnx_h1.json` records:

- CPU;
- 5 epochs;
- one seed `[42]`;
- H1 only;
- 154 nodes and 60,028 valid test cells;
- 153/154 mapped tickers, 23 groups, four singleton nodes.

The result is encouraging: sector QLIKE 1.8921 versus 1.9164 for statistical GAT and 1.9153 for
no-graph LSTM. However, one seed and five epochs do not establish convergence or stability. A large
cell count does not remove time/ticker dependence from inference.

The code correctly labels the run as a quick directional check at
`code/run_sector_ablation.py:114-117`. Keep that label and do not use this JSON as the final paper
table.

### S-05 — MEDIUM: this is static same-sector message passing, not hierarchical sector GAT

`sector_adjacency.py:36-47` creates only ticker-to-ticker same-label edges. There are no sector nodes,
sector embeddings, sector aggregates, or industry-to-sector hierarchy. The existing graph branch uses
last-timestep node features (`baselines/2026-08-21_har_anchored_residual/code/run_masked_rich.py:104-116`).

The accurate paper wording is **LSTM plus a static same-sector graph message-passing branch** or
**static sector-aware GAT**. With a fully connected sector clique, the second GAT layer does not
expand reach beyond the same sector; it mainly remixes the same block.

### S-06 — MEDIUM: metadata mapping is not fail-closed

`fetch_vn_sectors.py:38-47`, `fetch_sectors.py:36-49`, and `sector_adjacency.py:77-84` silently use
last-write-wins for duplicate tickers. Conflicting labels can alter the graph without an error.
`load_sector_map` strips whitespace but does not normalize case or dot/dash variants; normalization is
implemented only in selected fetch helpers.

**Action:** reject conflicting duplicate rows, validate normalized ticker identity, and record the
exact unmapped ticker list. Current aggregate coverage does not expose all metadata failures.

### S-07 — MEDIUM: result provenance and floor policy are incomplete

`run_sector_ablation.py:135-143` does not serialize:

- sector CSV SHA-256 and source revision;
- raw/processed data hash and git commit;
- sector level (`industry_name` versus `industry_code`);
- snapshot/effective-date policy;
- actual `top_k`, adjacency edge count/density, degree quantiles;
- unmapped ticker list;
- train/validation/test date boundaries and per-split valid-cell counts;
- QLIKE floor and output parameterization;
- actual/best epoch per seed.

This makes the JSON insufficient to reproduce or audit its QLIKE values independently.

**Action:** add these fields before promoting any sector result to paper evidence.

### S-08 — MEDIUM: tests do not certify real multi-seed training

`test/test_runner_and_fetch.py:104-137` monkeypatches `train_masked_rich`. The real-data smoke test
`test/test_smoke_forward.py:1-5,33-48` performs only one finite forward pass and explicitly no
training.

The tests do not verify actual convergence, early stopping, seed stability, ensemble key
intersection, floor behavior, DM numerical output, or identical `(ticker,date)` keys across all
variants. The reported 34 tests passing is therefore a plumbing result, not an end-to-end training
certificate.

Add at least one tiny real training integration test and assertions for identical test keys/masks,
split chronology, actual epochs, and no silent sample reduction.

### S-09 — MEDIUM: silent sample reduction is not guarded

The shared `_ens()` helper intersects prediction keys across seeds (`run_masked_rich.py:211-216`).
The sector runner reports `n_test_obs` from the no-graph result only and does not assert that every
seed and every variant has the same keys (`run_sector_ablation.py:126-143`).

Missing predictions can therefore be silently dropped from ensemble metrics and DM comparisons.

Add fail-closed assertions for equal key sets and record per-seed/per-variant counts before ensemble
aggregation.

### S-10 — MEDIUM/LOW: device and training metadata can be inaccurate

The result device is inferred from `CUDA_VISIBLE_DEVICES` at `run_sector_ablation.py:135-137`, while
the imported trainer selects the device using `torch.cuda.is_available()` at
`run_masked_rich.py:128-135`. Under an unusual environment, the result can report GPU while training
falls back to CPU.

Also, the result stores configured `cfg.epochs`, not actual/best epochs after early stopping
(`run_sector_ablation.py:135-143`, `run_masked_rich.py:171-192`). Record actual device and per-seed
training history.

### S-11 — LOW: coverage diagnostics are wrong for future capped runs

`sector_adj_for()` accepts `top_k` (`run_sector_ablation.py:80-84`), but `coverage()` always rebuilds
a fully connected graph (`sector_adjacency.py:87-95`). The current `top_k=None` result is consistent;
a future capped run would report degree/singleton statistics for a different graph.

Pass `top_k` into `coverage()` or calculate diagnostics from the actual adjacency matrix.

### S-12 — LOW: sector result is not wired into authoritative paper tables

The sector runner writes its own result schema (`run_sector_ablation.py:135-143`), while the paper
table generator has a fixed model list that does not include `sector_GAT`
(`scripts/paper/build_final_tables.py:24-28,122`). This is safe while the result is exploratory, but
it means there is no automated reconciliation if someone later presents it as a main model.

Either keep it explicitly outside the authoritative table or add a deliberate, validated paper-table
integration after the experiment passes the gates above.

## Agent consensus: passed checks

All agents agreed on these points:

- no obvious tensor-shape, self-loop, or invalid-source-mask defect in the sector-specific path;
- unmapped-ticker singleton behavior is conservative and correct;
- the three variants use shared panel, target masks, metrics, seed plumbing, and DM calls;
- current HNX coverage is high enough for the exploratory run;
- the remaining issues are principally experimental validity, metadata validity, and auditability.

## Fix order for the project AI

1. Fix `train_frac/val_frac` forwarding and add a regression test.
2. Add mapping conflict/normalization validation and immutable provenance.
3. Decide the point-in-time policy; remove “zero leakage” unless proven.
4. Run density-matched `top_k=5` plus a density sensitivity.
5. Add fail-closed key/mask equality and actual-training metadata.
6. Re-run full seeds/epochs/all horizons before any paper claim.

## Final statement for the paper team

The current code demonstrates a functioning **static sector-aware graph ablation**. It does not yet
demonstrate a leakage-free, capacity-controlled, stable improvement attributable specifically to
sector structure. The HNX H1 result may be reported internally as promising directional evidence, but
not as the final claim that sector GAT is the best model.
