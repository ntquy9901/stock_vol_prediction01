# Requirements — SERIAL LSTM→GNN hybrid (HNX volatility)

**Date:** 2026-08-30
**Paper anchor:** Sonani, Badii & Moin 2025 (arXiv:2502.15813) — §2.3 (LSTM temporal encoder feeding a
GNN relational layer) + §3.2 (combined linear Pearson-correlation + non-linear Apriori-lift edge).

## Goal
Build a NEW baseline whose network is a **SERIAL** LSTM→GNN hybrid: the LSTM produces a per-stock temporal
embedding `h_i` that becomes the GNN's **node feature**; the GNN then propagates `h_i` over a graph built
from BOTH linear (Pearson correlation) and non-linear (Apriori lift) edges. Measure whether this serial
architecture beats the plain no-graph LSTM on HNX h1 under the same folds/seeds/pipeline, and separate the
architecture effect from the graph-density effect.

## Why this is a distinct baseline (no duplication)
- **Delivered `MaskedRichNet` is PARALLEL** (CLAUDE.md): its GAT branch reads the RAW node features at day t
  (`x[:, :, -1, :]`), NOT the LSTM output. LSTM and GAT are two independent branches concatenated at the head.
- **This baseline is SERIAL:** the GNN's input IS the LSTM embedding `h_i` ("LSTM ra embedding rồi bỏ vào
  GNN"). The graph never sees the raw features directly; it only relabels/propagates temporal embeddings.
- Distinct from the separate in-flight hetero-GNN agent: that one keeps the two edge types SEPARATE with
  independent convolutions. This baseline uses the paper's COMBINED single graph (one undirected weighted
  adjacency; edge fires if `|ρ|>thr` OR `lift>thr`).

## Inputs
- Panel: **HNX** (screened universe, 162 tickers), horizon **h1**, lookback SEQ=10 (Config default).
- 5 node features `[parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20]` — the LSTM
  consumes the SEQ-window of these; identical to the delivered node vector (built by `masked_rich`).
- Combined corr+lift adjacency from `baselines/2026-08-29_corrlift_ablation/code/corrlift_edge.py`
  (READ-ONLY reuse), TRAIN-only (rows strictly before `D.d_va[0]`) then frozen.

## Outputs
- `results/lstm_gnn_serial_hybrid/lstm_gnn_serial_hybrid_hnx_h1.json` — per-variant metrics (MSE/RMSE/MAE/
  QLIKE/R²) ensemble + per-seed, date-clustered DM (QLIKE/SE/AE), train/val/test metrics + fit verdict +
  learning curves (over/under-fit evidence), edge density (dense AND paper thresholds).
- `docs/reports/2026-08-30_lstm_gnn_serial_hybrid.md` — metric table + DM + fit verdicts + edge density +
  architecture description + honest conclusion.

## Variants (controlled comparison, same folds/seeds, HNX h1)
1. `no_graph_LSTM` — the SERIAL model with `use_graph=False` (plain temporal baseline; head on `h` only).
2. `delivered_parallel_vol2pk` — the shipped PARALLEL `MaskedRichNet` (LSTM + GAT vol→PK), as CONTEXT.
3. `serial_hybrid_corrlift` — THIS baseline: SERIAL LSTM→GNN on the combined corr+lift graph.

## Edge thresholds
- Paper `|ρ|>0.7`, `lift>1.7` give a near-empty graph on thin HNX returns (measured in the corrlift baseline:
  `|ρ|>0.7` fires on 3/11781 pairs). To actually test the LSTM→GNN architecture (not re-confirm an empty
  graph) the primary graph uses **denser thresholds `ρ>0.25`, `lift>1.2`**. Report BOTH densities; explicitly
  note the paper thresholds are near-empty.

## Success / go-no-go criteria
- **Build correctness (hard gate):** unit tests prove (a) the GNN's input tensor IS the LSTM embedding (not
  the raw features), (b) mask-awareness (invalid nodes zeroed / excluded from attention), (c) correct output
  shape `[B, N]`, (d) the runner emits all metrics + DM + fit evidence on a tiny real HNX slice.
- **Statistical result (reported, not pass/fail):** DM(QLIKE) serial vs no_graph and serial vs delivered
  VolGA, with p-values + seed stability. Bar: 6+ prior graph probes were null on HNX h1 (no-graph LSTM
  QLIKE ≈ 1.81). **A null result is a valid, reportable outcome** — objective wording, no inflation.
- **Quality gate (hard):** pre-push gate green locally (C0 line=100% + C1 branch≥95% on changed lines,
  ruff --select F clean, lessons-regression, data-quality), 3-lens adversarial code review with
  critical/major fixed, then commit + push.

## Non-goals
- No new edge construction (reuse corrlift_edge read-only). No hyperparameter tuning. No multi-horizon /
  multi-panel sweep (HNX h1 only). No edit to any live-training-path file.
