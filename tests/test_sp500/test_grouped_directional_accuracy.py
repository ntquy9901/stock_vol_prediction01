"""Tests for src/common/evaluation.py::directional_accuracy_grouped /
evaluate_predictions_grouped — fixes cross-ticker diff contamination.

Bug: when pooling multiple tickers' predictions into one flat array before
calling directional_accuracy(), np.diff() at the boundary between ticker N's
last sample and ticker N+1's first sample computes a spurious "change"
between two unrelated tickers' volatility values. Found while debugging the
Phase 5 multi-horizon results (2026-08-01).

Test-first: written against the not-yet-existing function.
"""
import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


class TestDirectionalAccuracyGrouped:
    def test_excludes_cross_group_boundary_diff(self):
        """A boundary-crossing diff (naively computed on the pooled array)
        would flip the accuracy; the grouped version must not use it."""
        from src.common.evaluation import directional_accuracy_grouped

        # Ticker A: true goes down,down,up (2 diffs). Ticker B: true goes up,down (1 diff).
        true_a = np.array([3.0, 2.0, 1.0, 2.0])
        pred_a = np.array([3.0, 2.0, 1.0, 2.0])  # perfect direction match within A
        true_b = np.array([100.0, 200.0, 150.0])
        pred_b = np.array([100.0, 200.0, 150.0])  # perfect direction match within B

        acc = directional_accuracy_grouped([true_a, true_b], [pred_a, pred_b])

        # All within-group diffs match perfectly -> 100%, regardless of what
        # the (meaningless) A-to-B boundary "diff" would have been.
        assert acc == 100.0

    def test_naive_pooled_diff_would_be_worse_than_grouped(self):
        """Demonstrates the bug directly: pooling before np.diff() corrupts
        the result relative to the correct per-group computation.

        Constructed so every WITHIN-ticker direction matches (true and pred
        move the same way inside each group), but the spurious BOUNDARY diff
        (ticker A's last value -> ticker B's first value) has mismatched sign
        between true and pred -- purely an artifact of concatenation order,
        not a real prediction error.
        """
        from src.common.evaluation import directional_accuracy, directional_accuracy_grouped

        true_a = np.array([1.0, 2.0, 3.0])       # within A: up, up
        pred_a = np.array([10.0, 20.0, 30.0])    # within A: up, up (direction matches)
        true_b = np.array([1000.0, 999.0, 998.0])  # within B: down, down
        pred_b = np.array([5.0, 4.0, 3.0])          # within B: down, down (direction matches)

        grouped_acc = directional_accuracy_grouped([true_a, true_b], [pred_a, pred_b])
        assert grouped_acc == 100.0  # every real within-ticker diff matches

        pooled_true = np.concatenate([true_a, true_b])
        pooled_pred = np.concatenate([pred_a, pred_b])
        pooled_acc = directional_accuracy(pooled_true, pooled_pred)

        # Boundary: true jumps UP (3.0 -> 1000.0) but pred jumps DOWN (30.0 -> 5.0)
        # -- a spurious mismatch that only exists because of concatenation order.
        # 4 real diffs all correct + 1 spurious wrong boundary diff = 4/5 = 80%.
        assert abs(pooled_acc - 80.0) < 1e-9
        assert pooled_acc < grouped_acc

    def test_micro_averaged_across_uneven_group_sizes(self):
        from src.common.evaluation import directional_accuracy_grouped

        # Group 1: 1 diff, wrong. Group 2: 3 diffs, all correct.
        true_1, pred_1 = np.array([1.0, 2.0]), np.array([2.0, 1.0])  # true up, pred down -> wrong
        true_2 = np.array([5.0, 6.0, 5.0, 7.0])
        pred_2 = np.array([5.0, 6.0, 5.0, 7.0])  # matches exactly -> 3/3 correct

        acc = directional_accuracy_grouped([true_1, true_2], [pred_1, pred_2])
        # 3 correct out of 4 total diffs (1 wrong + 3 correct) = 75%
        assert abs(acc - 75.0) < 1e-9

    def test_skips_groups_too_short_for_a_diff(self):
        from src.common.evaluation import directional_accuracy_grouped

        true_short, pred_short = np.array([1.0]), np.array([1.0])  # 0 diffs, skipped
        true_ok = np.array([1.0, 2.0])
        pred_ok = np.array([1.0, 2.0])  # 1 correct diff

        acc = directional_accuracy_grouped([true_short, true_ok], [pred_short, pred_ok])
        assert acc == 100.0


class TestEvaluatePredictionsGrouped:
    def test_other_metrics_match_pooled_evaluate_predictions(self):
        """MSE/RMSE/MAE/R²/QLIKE are order-independent -> identical to pooling
        everything into evaluate_predictions()."""
        from src.common.evaluation import evaluate_predictions, evaluate_predictions_grouped

        true_a, pred_a = np.array([0.01, 0.02, 0.015]), np.array([0.011, 0.019, 0.016])
        true_b, pred_b = np.array([0.03, 0.025]), np.array([0.028, 0.026])

        pooled_metrics = evaluate_predictions(
            np.concatenate([true_a, true_b]), np.concatenate([pred_a, pred_b])
        )
        grouped_metrics = evaluate_predictions_grouped([true_a, true_b], [pred_a, pred_b])

        for key in ["mse", "rmse", "mae", "r2", "qlike"]:
            assert abs(pooled_metrics[key] - grouped_metrics[key]) < 1e-9

    def test_directional_accuracy_uses_grouped_computation(self):
        from src.common.evaluation import evaluate_predictions_grouped

        true_a, pred_a = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])
        true_b, pred_b = np.array([1000.0, 999.0]), np.array([1000.0, 998.0])

        metrics = evaluate_predictions_grouped([true_a, true_b], [pred_a, pred_b])
        # A: 2/2 correct. B: 1/1 correct (both go down). Total 3/3 = 100%.
        assert metrics["directional_accuracy"] == 100.0
