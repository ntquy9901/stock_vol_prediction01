"""Train-only preprocessing contracts for pooled price samples."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import SplitFrames, build_pooled_manifest, build_ticker_samples  # noqa: E402
from scaling import ArrayStandardizer, PreprocessorStore, TickerPreprocessor  # noqa: E402


def _frame(size: int, offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="D").strftime("%Y-%m-%d"),
            "parkinson_volatility": np.arange(size, dtype=float) + offset,
        }
    )


def _fit_store(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> PreprocessorStore:
    # val and test are deliberately accepted only as caller context: fitting sees train alone.
    del val, test
    return PreprocessorStore({0: TickerPreprocessor.fit(train, ["parkinson_volatility"], "parkinson_volatility")})


def test_val_and_test_cannot_change_train_parameters() -> None:
    train = _frame(70)
    first = _fit_store(train, _frame(30, 1.0), _frame(30, 2.0))
    second = _fit_store(train, _frame(30, 1e9), _frame(30, -1e9))

    assert first.to_dict() == second.to_dict()


def test_inverse_uses_explicit_ticker_id() -> None:
    first = TickerPreprocessor.fit(_frame(30, 10.0), ["parkinson_volatility"], "parkinson_volatility")
    second = TickerPreprocessor.fit(_frame(30, 100.0), ["parkinson_volatility"], "parkinson_volatility")
    store = PreprocessorStore({0: first, 1: second})

    got = store.inverse_targets(np.array([1, 0]), np.array([0.0, 0.0]))

    np.testing.assert_allclose(got, [114.5, 24.5])


def test_zero_variance_uses_unit_std_and_round_trips() -> None:
    scaler = ArrayStandardizer().fit(np.ones((4, 3)))

    np.testing.assert_allclose(scaler.std, np.ones(3))
    np.testing.assert_allclose(
        scaler.inverse_transform(scaler.transform(np.ones((4, 3)))),
        np.ones((4, 3)),
    )


def test_window_count_and_final_target_are_exact() -> None:
    samples = build_ticker_samples(_frame(70), "AAA", 0, seq_length=22, horizon=5)

    assert len(samples) == 70 - 22 - 5 + 1
    assert samples[-1].key.target_date == "2020-03-10"


def test_extreme_validation_and_test_values_cannot_change_train_bounds() -> None:
    normal = TickerPreprocessor.fit(_frame(70), ["parkinson_volatility"], "parkinson_volatility")
    transformed = normal.transform_frame(_frame(30, 1e9))

    assert transformed["y_model_raw"].max() == pytest.approx(normal.upper_bound)
    assert normal.to_dict() == TickerPreprocessor.fit(
        _frame(70), ["parkinson_volatility"], "parkinson_volatility"
    ).to_dict()


def test_model_target_is_clipped_but_evaluation_target_is_untouched() -> None:
    train = _frame(70)
    preprocessor = TickerPreprocessor.fit(train, ["parkinson_volatility"], "parkinson_volatility")
    test = _frame(70)
    test.loc[69, "parkinson_volatility"] = 1e6
    samples = build_ticker_samples(preprocessor.transform_frame(test), "AAA", 0, seq_length=22, horizon=5)

    assert samples[-1].y_model_raw == pytest.approx(preprocessor.upper_bound)
    assert samples[-1].y_eval_raw == 1e6


def test_manifest_is_deterministic_and_excludes_ineligible_tickers_once() -> None:
    eligible = {name: _frame(30, offset) for name, offset in zip(("train", "val", "test"), (0, 40, 80))}
    short = {name: _frame(26, offset) for name, offset in zip(("train", "val", "test"), (0, 40, 80))}
    split_frames = SplitFrames(
        frames={"ZZZ": eligible, "AAA": eligible, "BAD": short},
        ticker_to_id={"AAA": 0, "BAD": 1, "ZZZ": 2},
    )
    preprocessors = PreprocessorStore(
        {
            0: TickerPreprocessor.fit(eligible["train"], ["parkinson_volatility"], "parkinson_volatility"),
            1: TickerPreprocessor.fit(short["train"], ["parkinson_volatility"], "parkinson_volatility"),
            2: TickerPreprocessor.fit(eligible["train"], ["parkinson_volatility"], "parkinson_volatility"),
        }
    )

    manifest = build_pooled_manifest(split_frames, preprocessors, seq_length=22, horizon=5)

    assert manifest.exclusions == {"BAD": "insufficient windows in every split"}
    assert [(sample.key.target_date, sample.key.ticker_id) for sample in manifest.samples["train"]] == sorted(
        (sample.key.target_date, sample.key.ticker_id) for sample in manifest.samples["train"]
    )
