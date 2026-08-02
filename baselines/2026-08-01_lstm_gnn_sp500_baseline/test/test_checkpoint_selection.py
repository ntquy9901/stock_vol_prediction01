"""Test-first: checkpoint selection must track best-by-val-DirAcc, not
best-by-val-loss (normalized MSE).

Found via real training run (horizon=1, 10 epochs, seed=42): val_loss kept
decreasing every epoch (0.000013 -> 0.000000) even as val DirAcc collapsed
from 52-53% (epochs 1-5) to 0.01% (epochs 7-10) -- a classic prediction-
collapse overfit (train loss -> 0.000000 by epoch 7). Selecting "best" by
lowest val_loss picked a collapsed-epoch checkpoint instead of the
epoch-5 peak, because MSE loss is blind to directional accuracy.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.dirname(current_dir)
project_root = baseline_dir
for _ in range(2):
    project_root = os.path.dirname(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(baseline_dir, "code"))


class TestCheckpointSelection:
    def test_is_new_best_prefers_higher_dir_acc(self):
        from train_sp500_lstm_gnn import is_new_best

        assert is_new_best(current_dir_acc=53.0, best_dir_acc=52.0) is True
        assert is_new_best(current_dir_acc=51.0, best_dir_acc=52.0) is False
        assert is_new_best(current_dir_acc=52.0, best_dir_acc=52.0) is False

    def test_reproduces_the_real_collapse_scenario(self):
        """Mirrors the actual epoch-by-epoch val DirAcc sequence observed in
        the real horizon=1 run -- best-by-DirAcc must land on epoch 5
        (53.26%), not epoch 7-10 (0.01%, the collapsed checkpoints that
        best-by-loss would have picked since val_loss kept falling)."""
        from train_sp500_lstm_gnn import is_new_best

        val_dir_accs = [52.49, 49.88, 49.39, 53.12, 53.26, 47.75, 0.01, 0.01, 0.01, 0.01]

        best_epoch, best_dir_acc = None, float("-inf")
        for epoch, dir_acc in enumerate(val_dir_accs, start=1):
            if is_new_best(dir_acc, best_dir_acc):
                best_epoch, best_dir_acc = epoch, dir_acc

        assert best_epoch == 5
        assert best_dir_acc == 53.26
