# Masked availability-aware graph message passing — G0/G1 re-run

Date: 2026-08-08
Branch: `feature/masked-gnn` (worktree `.worktrees/masked-gnn`)
Scope: Add an availability-aware MASKED graph path to the pooled news GNN ablation and re-run
G0 vs G1 to test whether the graph-null result (G1 ≈ G0) was a data-scarcity artifact of the
26% common-date intersection. Source: `docs/reports/2026-08-08_gnn_sparse_data_research.md` Rank 1.

## What changed

| File | Purpose |
|---|---|
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/data.py` | `build_masked_graph_manifest` + `_masked_correlation_adjacency`; optional `presence_mask` on `GraphSnapshot`; presence key added to the snapshot hash only for masked snapshots. |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/models.py` | `_ResidualMessagePassing.forward` and `GraphAblationModel.forward` accept an optional `presence_mask`; present-only aggregation with absent-node invariance. |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/code/run_pilot.py` | `--graph masked\|intersection` switch (default intersection, unchanged); present-only loss (`_mean_snapshot_mse`) and raw-scale metrics; snapshot/date counts in output. |
| `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/test_masked_graph.py` | New test module (see below). |
| `baselines/.../requirements/masked_graph_requirements.md`, `design/masked_graph_design.md` | Spec + design note. |

### Method (Rank 1, no imputation)

Each pooled per-ticker window (already per-ticker chronological, train-only scaled, news-attached)
becomes one node of the graph snapshot for its own target date. Snapshots are padded to the fixed
33-ticker vocabulary; a ticker with no window on a date is masked (`presence_mask=0`) and never
imputed. Adjacency is the correlation over PRESENT nodes' price windows only. Grouping happens
inside each pooled split, so a snapshot never mixes splits and the graph-safe P3 boundary
(train targets ≤ `train_end_date`) is preserved. Message passing, loss, and raw-scale metrics run
over present nodes only; the frozen P3 encoder (no-grad), positivity floor, and nonpositive ≤ 1%
gate are unchanged. `--graph intersection` (default) keeps `presence_mask=None` and is byte-identical
to prior runs.

## Snapshot-count proof (data-scarcity confound removed)

| Regime | Distinct dates | Train snapshots | Val snapshots |
|---|---|---|---|
| Intersection (fixed 33-node) | ~1,296 (26% of 4,989) | ~900 | — |
| **Masked (variable node set)** | **4,941** | **4,523** | 1,237 |

The masked manifest uses 4,941 distinct trading dates (≈ the ~4,900-date union), 3.8× the 1,296
intersection dates, and 4,523 train snapshots vs ~900. Nonpositive-prediction fraction = 0.000 for
every masked G0/G1 run.

## Results (33 tickers, horizon 5, 5 epochs, CUDA)

Validation loss = normalized MSE (primary early-stopping/selection metric). Δ = G1 − G0
(negative ⇒ message passing helps).

Masked (this work), 3 seeds:

| Seed | G0 vloss | G1 vloss | Δ vloss | G0 QLIKE | G1 QLIKE | G0 RMSE | G1 RMSE | G0 DirAcc | G1 DirAcc |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.840138 | 0.842550 | +0.002411 | 0.50934 | 0.50822 | 0.001465 | 0.001458 | 48.85 | 49.09 |
| 123 | 0.839540 | 0.840302 | +0.000763 | 0.50860 | 0.50622 | 0.001464 | 0.001455 | 48.64 | 48.85 |
| 2026 | 0.839839 | 0.839782 | −0.000057 | 0.50875 | 0.50588 | 0.001464 | 0.001454 | 48.65 | 48.75 |
| mean | 0.839839 | 0.840878 | **+0.001039** | 0.50890 | 0.50677 | 0.001464 | 0.001456 | 48.71 | 48.90 |

Intersection (prior, batched, seed 42) — `results/pooled_news_gnn_g0g1_batched_2026-08-08_171457_seed42/`:

| Seed | G0 vloss | G1 vloss | Δ vloss | G0 QLIKE | G1 QLIKE | G0 RMSE | G1 RMSE | G0 DirAcc | G1 DirAcc |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.860710 | 0.862690 | +0.001980 | 0.68762 | 0.69628 | 0.002407 | 0.002410 | 48.61 | 48.52 |

## Verdict (honest)

With ~4× more graph data (4,523 train snapshots vs ~900; 4,941 dates vs 1,296), message passing
STILL does not help. On the primary validation loss the G1−G0 delta stays within ±0.0024 of zero
(G1 marginally worse on 2/3 seeds, marginally better on 1/3; mean +0.001). On the raw-scale metrics
the masked G1 is consistently a few 4th-decimal points better than masked G0 (QLIKE −0.002, RMSE
−0.00001, DirAcc +0.2 pp) — the sign flips relative to the intersection run (where G1 was slightly
worse on QLIKE/RMSE), but the magnitude in both regimes is negligible and not a lift.

Conclusion: the graph-null result is NOT an artifact of the intersection data scarcity. It is
robust — the cross-stock graph genuinely does not help VN30 volatility forecasting even with full
temporal depth and per-day node sets. The user's data-scarcity hypothesis is not confirmed; the
masked re-run removes the data-volume confound and turns the confounded null into a clean one
(research report outcome (a)).

Caveat on absolute levels: the masked ABSOLUTE metrics (QLIKE ≈ 0.509, RMSE ≈ 0.00146) are better
than the intersection ones (QLIKE ≈ 0.688, RMSE ≈ 0.00241), but both G0 and G1 improved equally —
this reflects the larger, different evaluation set and more training data for the frozen P3 encoder,
not a message-passing effect. The G1-vs-G0 gap (the ablation itself) stays null.

## Tests (RED → GREEN)

New `test_masked_graph.py` (written test-first; confirmed RED via `ImportError:
build_masked_graph_manifest`, then GREEN):
- (a) absent-node perturbation invariance at the layer and the `GraphAblationModel` level;
- (b) variable present-node count per snapshot;
- (c) present-only loss (`_mean_snapshot_mse`) and present-only evaluation records;
- (d) union-date recovery (synthetic: 15 union vs 5 intersection dates; real-data smoke: masked
  dates > 2× intersection and ≈ the longest ticker's history);
- regression: intersection snapshots keep `presence_mask=None`.

Commands run (real output):
- `python -m pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/ -q` → **134 passed**.
- `python -m ruff check baselines/.../code/ baselines/.../test/test_masked_graph.py` → **All checks passed!**
- Data-quality gate (shared git-tracked data): `check_schema()` → **SCHEMA PASS (34/34 artifacts valid)**;
  `check_drift()` → **DRIFT INFO** (Evidently DataDrift report generated on ACB_processed.csv,
  ref=2800/cur=1200 rows).

Coverage (C0=100% / C1≥80% via `diff-cover`): **Not run** — diff-cover/pytest-cov not set up in this
repo (documented tooling gap in CLAUDE.md). New behavior is covered by the tests above.

## Code review

**Not run via `/code-review` yet** — parent will verify/push and a pre-push hook runs the gate.
Self-review checklist applied: no cross-baseline edits; intersection path byte-unchanged
(`presence_mask=None`, presence key omitted from its hash); no imputation / no future information;
per-ticker split, train-only scalers, graph-safe boundary, positivity floor, nonpositive gate all
preserved; frozen encoders receive no gradients (existing assertion still holds).

## Output locations

- `results/pooled_news_gnn_masked_g0g1_2026-08-08_212959_seed42/h5/{G0,G1}/results.json`
- `results/pooled_news_gnn_masked_g0g1_2026-08-08_214227_seed123/h5/{G0,G1}/results.json`
- `results/pooled_news_gnn_masked_g0g1_2026-08-08_214916_seed2026/h5/{G0,G1}/results.json`
- Each run dir also holds `graph_validation_comparison.json` (mode, snapshot/date counts, paired delta).

## Follow-ups / risks

- 5-epoch screening only (per training policy); a longer run is not warranted given the robust null.
- Rank 2 (long-history panel-selection ablation) remains available as a triangulating diagnostic
  if a further check is wanted; it carries a disclosed survivorship/look-ahead caveat.
- Paper Limitations can now state the stronger, unconfounded claim (research report outcome (a)).
