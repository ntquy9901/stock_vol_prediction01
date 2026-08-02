# Design: LSTM-GNN Baseline for S&P 500

**Date:** 2026-08-01

## 1. Data flow

```
data/processed_sp500/{30 tickers}_processed.csv   (this session's process_parkinson_pipeline run)
        │ MultiStockDataset(data_dir='data/processed_sp500', ...)   [UNCHANGED, src/lstm_gat_hybrid/dataset.py]
        │   - generates HAR features per ticker internally
        │   - builds per-sequence k-NN/correlation graph (DynamicGraphBuilder)
        │   - per-ticker normalization (StandardScaler-based, VolatilityNormalizer)
        │   - temporal split 70/15/15
        ▼
create_multi_stock_dataloaders(...)  → train/val/test DataLoader
        │
        ▼
ParallelLSTMGNN(config)  [UNCHANGED, src/lstm_gat_hybrid/model_parallel.py]
   LSTM stream (per-ticker temporal) + GAT stream (per-day spatial) → concat → MLP → prediction
        │
        ▼
code/train_sp500_lstm_gnn.py (NEW, this baseline)  — training loop, adapted from
src/lstm_gat_hybrid/train_parallel.py (same reuse pattern, only data_dir + horizon changed)
```

## 2. Config choices

- `config.num_stocks = 30` — already the class default (matches VN30's 30-stock count and this
  session's 30-ticker S&P 500 selection). No override needed.
- `config.forecast_horizon` — set from `--forecast_horizon` CLI arg (1/5/10), passed through to
  `create_multi_stock_dataloaders(forecast_horizon=...)`.
- `config.num_epochs` — capped at 10 for this experimentation phase (CLAUDE.md Training Policy),
  NOT the paper's 40-70 default in `train_parallel.py` — this is intentionally a smaller run to fit
  the same 10-epoch budget used throughout Phase 3-5, revisit if results look promising.
- `graph_method`, `lstm_hidden_dim`, `gat_hidden_dim`, dropout, etc. — left at `LSTMGATConfig` defaults
  (VN30-proven values), not re-tuned for S&P 500 in this first pass.

## 3. Simplicity / Anti-Abstraction gates

- **Simplicity Gate: PASS.** One new file (`train_sp500_lstm_gnn.py`), no new classes, no new
  abstraction — a thin script that calls existing factory functions with different arguments.
- **Anti-Abstraction Gate: PASS.** Directly reuses `ParallelLSTMGNN`, `MultiStockDataset`,
  `LSTMGATConfig`, `evaluate_predictions` — no wrapper layer added around any of them.

## 4. Known cost/risk

- Per-sequence graph construction (`DynamicGraphBuilder` inside `MultiStockDataset._create_sequences`)
  recomputes a k-NN graph for EVERY window at dataset-build time (not per training step) — O(num_stocks²)
  per window, done once when the dataset is constructed, not repeated every epoch. For 30 stocks ×
  ~2600 windows (70% of ~3790 rows, minus seq_length) this is a known, bounded, one-time cost — not
  re-verified against a runtime budget in this design; if dataset construction is too slow, that's a
  concrete signal to revisit (not assumed away).
- CPU-only training (no GPU available in this session's environment, per `config.device = 'cpu'`
  default) — 10 epochs on 30 stocks may be slower than the 3-ticker plain-LSTM baselines; monitored
  during the actual run, not pre-optimized.

## 5. Test plan

`test/test_sp500_data_loads.py`: instantiate `MultiStockDataset(data_dir='data/processed_sp500', ...)`
(or a temp-dir fixture with a small synthetic 3-ticker subset to keep the test fast) and assert:
- exactly the expected tickers are loaded (no VN30 contamination, no `processing_summary` picked up
  as a fake "ticker").
- at least 1 sequence is produced with the expected shapes (`x: [seq_len, num_stocks, 3]`,
  `adj_matrix: [num_stocks, num_stocks]`, `y: [num_stocks]`).
