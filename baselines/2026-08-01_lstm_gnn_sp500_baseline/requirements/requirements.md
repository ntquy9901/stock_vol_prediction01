# Requirements: LSTM-GNN Baseline for S&P 500 (horizon 1/5/10)

**Date:** 2026-08-01
**Branch:** `global-benchmark`

---

## 1. Goal

Port VN30's proven `ParallelLSTMGNN` architecture (LSTM temporal + GAT spatial, concat fusion —
`src/lstm_gat_hybrid/`) to S&P 500, using the 30-ticker subset just HAR-processed
(`data/processed_sp500/`, see this session's process_parkinson_pipeline run), for horizons {1, 5, 10}.
This replaces the plain per-ticker LSTM used in Phase 3/5 (`train_enhanced.py`) with a model that
actually captures cross-stock relationships, matching VN30's best-performing architecture.

## 2. Why this should be a thin wrapper, not new model code

Studied `src/lstm_gat_hybrid/` before writing anything (per CLAUDE.md's "study proven solution first"
rule, §"LSTM-GNN Normalization Failure" lesson):
- `ParallelLSTMGNN.forward()` reads shapes from the actual input tensor (`batch, seq_len, num_stocks,
  num_features = x.shape`), not from `config.num_stocks` — genuinely shape-agnostic.
- `MultiStockDataset._load_multi_stock_data(data_dir)` globs `*.csv` in whatever directory it's given,
  extracts ticker from filename, skips anything missing `date`/`parkinson_volatility` columns (so
  `processing_summary.csv` in `data/processed_sp500/` is safely skipped, not a schema mismatch).
- `create_multi_stock_dataloaders(data_dir=..., forecast_horizon=...)` already parameterizes both the
  data directory and the horizon — no source change needed for either.

**Conclusion: reuse `src/lstm_gat_hybrid/{model_parallel,dataset,config}.py` completely unchanged
(read-only import, per CLAUDE.md §3.F.3 hard isolation)**, write ONE new training script here that
points `data_dir` at `data/processed_sp500` and exposes `--forecast_horizon`.

## 3. Acceptance criteria

- [ ] `code/train_sp500_lstm_gnn.py` trains `ParallelLSTMGNN` on the 30 processed S&P 500 tickers,
      `--forecast_horizon {1,5,10}`, reusing `src/lstm_gat_hybrid/` unmodified.
- [ ] Smoke test (2 epochs) passes before any 10-epoch run.
- [ ] 3 runs (horizon 1, 5, 10), 10 epochs each (Training Policy cap), results saved with all 6
      mandatory metrics.
- [ ] `test/` has at least 1 test verifying the S&P 500 data loads into `MultiStockDataset` correctly
      (30 tickers found, no VN30 contamination).
- [ ] Code review (self, adversarial) before done.

## 4. Scope

### In scope
- New training script (this baseline folder only).
- 30-ticker S&P 500 dataset already prepared this session (`data/processed_sp500/`).
- Horizons 1, 5, 10 — matches Phase 5's scope, extends it to the GNN architecture.

### Out of scope
- Any modification to `src/lstm_gat_hybrid/` (hard isolation, per CLAUDE.md §3.F.3).
- News/sentiment fusion (VN30's later baselines) — HAR-only GNN first, matching VN30's own
  incremental history (HAR-only backbone came before news fusion).
- Expanding beyond 30 tickers.
