#!/usr/bin/env bash
# Variant 1: price-only backbone (train_parallel_enhanced.py, knn). Resume each 20-ep seed +20 -> 40.
set -e
cd /c/luanvan/stock_vol_prediction01
export PYTHONUNBUFFERED=1
S="src/lstm_gat_hybrid/train_parallel_enhanced.py"

python $S --graph_method knn --seed 42 --epochs 20 \
  --resume_checkpoint results/parallel_lstm_gnn_knn_2026-08-03_230722/best_parallel_model.pth \
  --resume_results_dir results/parallel_lstm_gnn_knn_2026-08-03_230722

python $S --graph_method knn --seed 123 --epochs 20 \
  --resume_checkpoint results/parallel_lstm_gnn_knn_seed123_2026-08-03_234613/best_parallel_model.pth \
  --resume_results_dir results/parallel_lstm_gnn_knn_seed123_2026-08-03_234613

python $S --graph_method knn --seed 2026 --epochs 20 \
  --resume_checkpoint results/parallel_lstm_gnn_knn_seed2026_2026-08-04_000327/best_parallel_model.pth \
  --resume_results_dir results/parallel_lstm_gnn_knn_seed2026_2026-08-04_000327

echo "BACKBONE_ALL_DONE"
