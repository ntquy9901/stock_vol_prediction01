"""Tests for the 20-epoch pooled pilot aggregation helpers (TDD: written first)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from scipy import stats

from aggregate import aggregate_metrics, load_seed_comparison, paired_t

_METRICS = ("mse", "rmse", "mae", "r2", "qlike", "directional_accuracy")


def _row(name: str, base: float) -> dict[str, float]:
    return {"config_name": name, **{metric: base + index for index, metric in enumerate(_METRICS)}}


def test_load_seed_comparison_reads_rows_by_config(tmp_path: Path) -> None:
    payload = {"rows": [_row("P0", 1.0), _row("P1", 2.0), _row("P2", 3.0), _row("P3", 4.0)]}
    path = tmp_path / "validation_comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_seed_comparison(path)
    assert set(loaded) == {"P0", "P1", "P2", "P3"}
    assert loaded["P1"]["mse"] == 2.0
    assert loaded["P3"]["directional_accuracy"] == 4.0 + (len(_METRICS) - 1)


def test_load_seed_comparison_rejects_missing_config(tmp_path: Path) -> None:
    payload = {"rows": [_row("P0", 1.0)]}
    path = tmp_path / "validation_comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_seed_comparison(path)


def test_aggregate_metrics_mean_and_sample_std() -> None:
    seeds = [
        {"P1": {"rmse": 1.0}},
        {"P1": {"rmse": 2.0}},
        {"P1": {"rmse": 3.0}},
    ]
    aggregated = aggregate_metrics(seeds, configs=("P1",), metrics=("rmse",))
    mean, std = aggregated["P1"]["rmse"]
    assert mean == pytest.approx(2.0)
    # sample std (ddof=1) of [1,2,3] == 1.0
    assert std == pytest.approx(1.0)


def test_paired_t_matches_scipy_ttest_rel() -> None:
    a = [0.51, 0.52, 0.50]
    b = [0.48, 0.49, 0.47]
    t_stat, p_value, mean_diff = paired_t(a, b)
    expected = stats.ttest_rel(a, b)
    assert t_stat == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue)
    assert mean_diff == pytest.approx(sum(x - y for x, y in zip(a, b)) / 3)


def test_paired_t_zero_difference_is_nan_or_zero() -> None:
    a = [1.0, 2.0, 3.0]
    t_stat, p_value, mean_diff = paired_t(a, a)
    assert mean_diff == pytest.approx(0.0)
    assert math.isnan(t_stat) or t_stat == pytest.approx(0.0)
