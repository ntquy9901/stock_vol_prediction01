#!/usr/bin/env bash
# Variant 3: no-graph ablation (identity adjacency). Resume each 20-ep seed +20 -> 40.
set -e
cd /c/luanvan/stock_vol_prediction01
export PYTHONUNBUFFERED=1
S="scripts/ablation_no_graph/run_no_graph_ablation.py"

python $S --seeds 42 \
  --resume_checkpoint results/no_graph_ablation_seed42_2026-08-05_225806/best_no_graph_model.pth \
  --resume_results_dir results/no_graph_ablation_seed42_2026-08-05_225806

python $S --seeds 123 \
  --resume_checkpoint results/no_graph_ablation_seed123_2026-08-05_231327/best_no_graph_model.pth \
  --resume_results_dir results/no_graph_ablation_seed123_2026-08-05_231327

python $S --seeds 2026 \
  --resume_checkpoint results/no_graph_ablation_seed2026_2026-08-05_232845/best_no_graph_model.pth \
  --resume_results_dir results/no_graph_ablation_seed2026_2026-08-05_232845

echo "NO_GRAPH_ALL_DONE"
