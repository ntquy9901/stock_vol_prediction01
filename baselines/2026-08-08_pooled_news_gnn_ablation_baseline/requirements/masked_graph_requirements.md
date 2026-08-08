# Masked availability-aware graph — requirements

## Goal

Resolve whether the graph-null result (G1 ≈ G0) in the intersection graph ablation was a
data-scarcity artifact of the 26% common-date intersection (1,296 of 4,989 dates), by re-running
G0 vs G1 on an availability-aware MASKED graph that trains on the full ~4,900-date union.

## Input / output

- Input: existing `data/processed/*_processed.csv` (33 tickers), `dual_group_news_panel.parquet`
  (+ provenance), the graph-safe P3 checkpoint pipeline.
- Output: masked G0 (message passing OFF) and G1 (ON) validation metrics (6 mandatory metrics),
  nonpositive-prediction fraction, and the number of dates/snapshots used, in distinct output dirs;
  a comparison against the intersection G0/G1.

## Acceptance criteria

- The masked manifest uses far more dates than the intersection: distinct snapshot dates ≫ 1,296,
  near the ~4,900-date union.
- Absent tickers on a date are masked, never imputed. No future information used.
- An absent node's features do not influence any present node's output (perturbation invariance).
- Variable present-node count per snapshot is handled.
- Loss and metrics aggregate over present nodes only.
- Per-ticker chronological split, train-only scalers, graph-safe P3 boundary (train targets ≤
  boundary), shuffle=False, seeds, provenance all preserved.
- Frozen P3 encoder (no-grad), denormalized positivity floor, and nonpositive ≤ 1% gate preserved.
- `--graph masked|intersection` switch; default `intersection` unchanged (old results reproducible).

## Go / no-go verdict

- If masked G1 < masked G0 (message passing now helps with ~4× more data) → the intersection null
  was a data-scarcity artifact (hypothesis confirmed).
- If masked G1 ≈ masked G0 (still null) → the graph genuinely does not help even with full data;
  the null is robust, not a confound. Report honestly either way.
