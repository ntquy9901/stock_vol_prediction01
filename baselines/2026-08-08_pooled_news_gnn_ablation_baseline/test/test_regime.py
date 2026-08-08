"""A1 ablation: common-date-restricted pooled regime and its leakage guards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CODE = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
for _path in (str(_ROOT), str(_CODE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import (  # noqa: E402
    PooledManifest,
    PooledSample,
    SampleKey,
    SplitFrames,
    chronological_split,
    common_trading_dates,
    restrict_train_to_common_dates,
)
from scaling import PreprocessorStore, TickerPreprocessor  # noqa: E402
import run_pilot  # noqa: E402


def _store(frames: dict[str, pd.DataFrame]) -> PreprocessorStore:
    return PreprocessorStore({
        index: TickerPreprocessor.fit(frame, ["parkinson_volatility"], "parkinson_volatility")
        for index, (_ticker, frame) in enumerate(sorted(frames.items()))
    })


def _transformed_dates(store: PreprocessorStore, ticker_id: int, frame: pd.DataFrame) -> set[str]:
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return set(store.get(ticker_id).transform_frame(normalized)["date"])


def test_common_trading_dates_is_post_har_intersection() -> None:
    dates = pd.date_range("2020-01-01", periods=80, freq="B").strftime("%Y-%m-%d")
    aaa = pd.DataFrame({"date": dates, "parkinson_volatility": np.linspace(1.0, 2.0, 80)})
    bbb = aaa.drop(index=40).reset_index(drop=True)  # BBB missing one middle trading day
    frames = {"AAA": aaa, "BBB": bbb}
    store = _store(frames)

    common = common_trading_dates(frames, store)

    expected = sorted(_transformed_dates(store, 0, aaa) & _transformed_dates(store, 1, bbb))
    assert list(common) == expected
    # The 21-day HAR warm-up is excluded from the common axis for every ticker.
    assert dates[0] not in common
    assert dates[21] in common
    # A day one ticker never traded cannot be a globally-common date.
    assert dates[40] not in common


def _axis_sample(ticker_id: int, target_date: str, input_dates: list[str]) -> PooledSample:
    length = len(input_dates)
    return PooledSample(
        SampleKey(ticker_id, "AAA", target_date),
        np.ones((length, 1)),
        np.zeros((length, 0)),
        np.zeros(length, dtype=np.int8),
        1.0,
        1.0,
        1.0,
        tuple(input_dates),
    )


def test_restrict_train_keeps_only_in_axis_windows_and_leaves_eval_untouched() -> None:
    common = {"2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"}
    keep = _axis_sample(0, "2020-01-07", ["2020-01-02", "2020-01-03"])
    drop_input = _axis_sample(0, "2020-01-07", ["2020-01-01", "2020-01-03"])  # input outside axis
    drop_target = _axis_sample(0, "2020-01-08", ["2020-01-02", "2020-01-03"])  # target outside axis
    # val/test intentionally reference out-of-axis dates: they must be preserved verbatim.
    val = _axis_sample(0, "2020-01-20", ["2020-01-19"])
    test = _axis_sample(0, "2020-01-30", ["2020-01-29"])
    manifest = PooledManifest(
        {"train": (keep, drop_input, drop_target), "val": (val,), "test": (test,)},
        {}, {"AAA": 0}, "pre",
    )

    restricted = restrict_train_to_common_dates(manifest, common)

    assert restricted.samples["train"] == (keep,)
    # Held-out splits are the full pooled sets -- identical evaluation set across regimes.
    assert restricted.samples["val"] == manifest.samples["val"]
    assert restricted.samples["test"] == manifest.samples["test"]
    assert restricted.ticker_to_id == manifest.ticker_to_id
    assert restricted.preprocessing_hash == manifest.preprocessing_hash


def test_restrict_train_raises_when_train_is_emptied() -> None:
    common = {"2020-01-02"}
    manifest = PooledManifest(
        {"train": (_axis_sample(0, "2020-01-09", ["2020-01-09"]),),
         "val": (_axis_sample(0, "2020-01-02", ["2020-01-02"]),),
         "test": (_axis_sample(0, "2020-01-02", ["2020-01-02"]),)},
        {}, {"AAA": 0}, "pre",
    )

    with pytest.raises(ValueError, match="common-date"):
        restrict_train_to_common_dates(manifest, common)


def _two_ticker_splits() -> SplitFrames:
    dates = pd.date_range("2018-01-01", periods=420, freq="B")
    aaa_full = pd.DataFrame({"date": dates[:400], "parkinson_volatility": np.linspace(1.0, 3.0, 400)})
    bbb_full = pd.DataFrame({"date": dates[20:420], "parkinson_volatility": np.linspace(2.0, 4.0, 400)})
    frames = {ticker: chronological_split(frame) for ticker, frame in {"AAA": aaa_full, "BBB": bbb_full}.items()}
    return SplitFrames(frames, {"AAA": 0, "BBB": 1})


def _sample_keys(manifest: PooledManifest, split: str) -> set[tuple[int, str]]:
    return {(sample.key.ticker_id, sample.key.target_date) for sample in manifest.samples[split]}


def test_common_date_regime_reuses_scalers_and_reduces_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    splits = _two_ticker_splits()
    monkeypatch.setattr(run_pilot, "load_and_split_price_data", lambda _path: splits)

    pooled = run_pilot.build_screening_inputs(
        smoke=False, max_tickers=2, phase="P1", regime="pooled"
    )
    common = run_pilot.build_screening_inputs(
        smoke=False, max_tickers=2, phase="P1", regime="common-date"
    )

    # Leakage guard: the common-date regime MUST reuse the pooled per-ticker
    # train-fitted scalers/winsor bounds -- no refit on the smaller subset.
    assert common.store.to_dict() == pooled.store.to_dict()
    # Only the training-sample SET changes; common-date train is a strict subset.
    pooled_train = _sample_keys(pooled.manifest, "train")
    common_train = _sample_keys(common.manifest, "train")
    assert common_train < pooled_train
    # Held-out splits are identical across regimes -> identical evaluation set.
    for split in ("val", "test"):
        assert common.manifest.content_hash(split) == pooled.manifest.content_hash(split)
    assert common.smoke_filter["regime"] == "common-date"


def test_parse_args_regime_defaults_to_pooled_and_rejects_unknown() -> None:
    assert run_pilot.parse_args([]).regime == "pooled"
    assert run_pilot.parse_args(["--regime", "common-date"]).regime == "common-date"
    with pytest.raises(SystemExit):
        run_pilot.parse_args(["--regime", "bogus"])


def test_parse_args_rejects_common_date_regime_for_graph_phase() -> None:
    assert run_pilot.parse_args(["--phase", "graph"]).regime == "pooled"
    with pytest.raises(SystemExit):
        run_pilot.parse_args(["--phase", "graph", "--regime", "common-date"])


def test_common_date_news_path_is_byte_identical_to_pooled_per_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data import NewsPanel  # noqa: PLC0415

    splits = _two_ticker_splits()
    # One real news key on an early AAA trading day plus an all-zero elsewhere: the point is
    # that whatever news a surviving sample carries must match the pooled sample of the same key.
    panel = NewsPanel({("AAA", "2018-03-01"): np.array([1.0, 2.0])}, ("f0", "f1"), {})
    monkeypatch.setattr(run_pilot, "load_and_split_price_data", lambda _path: splits)
    monkeypatch.setattr(run_pilot, "load_runner_news_panel", lambda *_a, **_k: panel)

    pooled = run_pilot.build_screening_inputs(smoke=False, max_tickers=2, phase="P2", regime="pooled")
    common = run_pilot.build_screening_inputs(smoke=False, max_tickers=2, phase="P2", regime="common-date")

    assert common.store.to_dict() == pooled.store.to_dict()
    pooled_by_key = {
        (s.key.ticker_id, s.key.target_date): s for split in ("train", "val", "test")
        for s in pooled.manifest.samples[split]
    }
    common_count = 0
    for split in ("train", "val", "test"):
        for sample in common.manifest.samples[split]:
            twin = pooled_by_key[(sample.key.ticker_id, sample.key.target_date)]
            assert sample.x_news.shape[-1] == 2  # news width preserved, not dropped
            np.testing.assert_array_equal(sample.x_news, twin.x_news)
            np.testing.assert_array_equal(sample.news_mask, twin.news_mask)
            common_count += 1
    assert common_count > 0


def _tiny_store() -> PreprocessorStore:
    scaler = TickerPreprocessor(
        ("parkinson_volatility", "har_weekly", "har_monthly"), "parkinson_volatility", 0.0, 2.0,
        _identity_scaler(3), _identity_scaler(1),
    )
    return PreprocessorStore({0: scaler})


def _identity_scaler(width: int):
    from scaling import ArrayStandardizer  # noqa: PLC0415

    return ArrayStandardizer(np.zeros(width), np.ones(width))


def _tiny_manifest() -> PooledManifest:
    def sample(index: int) -> PooledSample:
        value = float(index + 1)
        return PooledSample(
            SampleKey(0, "AAA", f"2020-02-{index + 1:02d}"),
            np.tile([value, value + 1, value + 2], (22, 1)),
            np.zeros((22, 0)), np.zeros(22, dtype=np.int8), value + 10, value + 10, value + 10,
            tuple(f"2020-01-{day:02d}" for day in range(1, 23)),
        )

    return PooledManifest(
        {"train": tuple(sample(index) for index in range(4)),
         "val": tuple(sample(index) for index in range(4, 7)), "test": ()},
        {}, {"AAA": 0}, "preprocessing",
    )


def test_run_pooled_screening_records_regime_in_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from torch.utils.data import DataLoader  # noqa: PLC0415

    manifest = _tiny_manifest()
    loaders = {
        split: DataLoader(run_pilot._ManifestDataset(manifest.samples[split], _tiny_store()), batch_size=2,
                          shuffle=False)
        for split in ("train", "val")
    }

    def build_inputs(smoke, max_tickers, phase, horizon=5, regime="pooled"):
        return run_pilot.ScreeningInputs(manifest, _tiny_store(), loaders, {"regime": regime})

    def train(*args, **_kwargs):
        output_dir = Path(args[3])
        output_dir.mkdir(parents=True, exist_ok=True)
        result = output_dir / "results.json"
        result.write_text(json.dumps({"validation_metrics": {
            "mse": 1.0, "rmse": 1.0, "mae": 1.0, "r2": 0.0, "qlike": 1.0, "directional_accuracy": 50.0,
        }}), encoding="utf-8")
        return result

    monkeypatch.setattr(run_pilot, "build_screening_inputs", build_inputs)
    monkeypatch.setattr(run_pilot, "run_training", train)
    args = run_pilot.parse_args([
        "--phase", "P1", "--epochs", "1", "--output-dir", str(tmp_path), "--regime", "common-date",
    ])

    run_pilot.run_pooled_screening(args)

    metadata = json.loads((tmp_path / "h5" / "screening_metadata.json").read_text(encoding="utf-8"))
    assert metadata["regime"] == "common-date"
