"""Contract tests for validated raw per-ticker price splits."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = _ROOT / "baselines" / "2026-08-08_pooled_news_gnn_ablation_baseline" / "code"
_NEWS_HELPER_DIR = _ROOT / "baselines" / "2026-07-25_dual_group_news_embedding_baseline" / "code"
for _path in (str(_ROOT), str(_CODE_DIR), str(_NEWS_HELPER_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data import (  # noqa: E402
    NewsPanel,
    PooledManifest,
    PooledSample,
    SampleKey,
    attach_news,
    chronological_split,
    effective_trading_date,
    load_effective_news_panel,
    load_and_split_price_data,
)


def _frame(size: int, ticker: str = "AAA") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=size, freq="B").strftime("%Y-%m-%d"),
            "parkinson_volatility": np.linspace(0.01, 0.02, size),
            "ticker": ticker,
        }
    )


def _frame_with_duplicate_date() -> pd.DataFrame:
    frame = _frame(100)
    frame.loc[99, "date"] = frame.loc[98, "date"]
    return frame


def test_split_is_per_ticker_chronological_and_disjoint() -> None:
    parts = chronological_split(_frame(100))

    assert [len(parts[key]) for key in ("train", "val", "test")] == [70, 15, 15]
    assert parts["train"].date.max() < parts["val"].date.min()
    assert parts["val"].date.max() < parts["test"].date.min()


def test_invalid_duplicate_date_fails_before_split() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        chronological_split(_frame_with_duplicate_date())


def test_non_monotonic_dates_fail_before_split() -> None:
    frame = _frame(100)
    frame.loc[[1, 2], "date"] = frame.loc[[2, 1], "date"].to_numpy()

    with pytest.raises(ValueError, match="monotonic"):
        chronological_split(frame)


def test_raw_records_are_frozen_and_preserve_values() -> None:
    key = SampleKey(ticker_id=4, ticker="AAA", target_date="2021-01-08")
    sample = PooledSample(
        key=key,
        x_price_raw=np.array([[0.1]]),
        x_news=np.array([[0.0]]),
        news_mask=np.array([0]),
        y_raw=0.2,
    )

    assert sample.key == key
    assert sample.y_raw == 0.2
    with pytest.raises(Exception):
        key.ticker = "ZZZ"  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_price_raw": np.ones(3)},
        {"x_price_raw": np.ones((2, 1)), "x_news": np.ones((3, 1))},
        {"news_mask": np.array([0, 2])},
        {"news_mask": np.array([0]), "y_raw": np.nan},
    ],
)
def test_pooled_sample_rejects_invalid_shapes_masks_and_targets(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "key": SampleKey(0, "AAA", "2020-01-10"),
        "x_price_raw": np.ones((2, 1)),
        "x_news": np.ones((2, 1)),
        "news_mask": np.array([0, 1]),
        "y_raw": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        PooledSample(**values)  # type: ignore[arg-type]


def test_ticker_ids_are_sorted_and_stable(tmp_path: Path) -> None:
    _frame(100, ticker="ZZZ").drop(columns="ticker").to_csv(tmp_path / "ZZZ_processed.csv", index=False)
    _frame(100, ticker="AAA").drop(columns="ticker").to_csv(tmp_path / "AAA_processed.csv", index=False)

    splits = load_and_split_price_data(tmp_path)

    assert splits.ticker_to_id == {"AAA": 0, "ZZZ": 1}
    assert set(splits.frames) == {"AAA", "ZZZ"}
    assert [len(splits.frames["AAA"][key]) for key in ("train", "val", "test")] == [70, 15, 15]


def _write_news_panel(tmp_path: Path, rows: list[tuple[str, str, list[float]]]) -> Path:
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        [(ticker, date, *features) for ticker, date, features in rows],
        columns=["ticker", "date", "f1", "f0"],
    ).to_parquet(path, index=False)
    return path


def _sample_with_dates(ticker: str, dates: list[str], y: float = 1.0, mask: list[int] | None = None) -> PooledSample:
    return PooledSample(
        key=SampleKey(0, ticker, "2020-01-10"),
        x_price_raw=np.ones((len(dates), 1)),
        x_news=np.zeros((len(dates), 0)),
        news_mask=np.zeros(len(dates), dtype=np.int8) if mask is None else np.asarray(mask, dtype=np.int8),
        y_raw=y,
        input_dates=tuple(dates),
    )


def test_news_panel_uses_effective_dates_and_masks_missing_days(tmp_path: Path) -> None:
    panel_path = _write_news_panel(tmp_path, [("AAA", "2020-01-02", [2.0, 1.0])])
    sample = _sample_with_dates("AAA", ["2020-01-01", "2020-01-02"])

    panel = load_effective_news_panel(panel_path)
    attached = attach_news([sample], panel, ["f0", "f1"])[0]

    assert panel.feature_cols == ("f0", "f1")
    np.testing.assert_array_equal(attached.news_mask, [0, 1])
    np.testing.assert_allclose(attached.x_news, [[0, 0], [1, 2]])
    assert attached.x_news.dtype == np.float32
    assert not attached.x_news.flags.writeable


def test_after_close_timestamp_maps_to_next_effective_trading_date() -> None:
    exact = effective_trading_date(
        pd.Series(["2020-01-02T15:00:00+07:00"]),
        pd.Series(["2020-01-02", "2020-01-03"]),
    ).iloc[0]
    after = effective_trading_date(
        pd.Series(["2020-01-02T15:00:01+07:00"]),
        pd.Series(["2020-01-02", "2020-01-03"]),
    ).iloc[0]

    assert exact.strftime("%Y-%m-%d") == "2020-01-02"
    assert after.strftime("%Y-%m-%d") == "2020-01-03"


def test_news_panel_rejects_duplicate_nonfinite_and_mismatched_features(tmp_path: Path) -> None:
    duplicate = _write_news_panel(tmp_path, [("AAA", "2020-01-02", [1.0, 2.0]), ("AAA", "2020-01-02", [3.0, 4.0])])
    with pytest.raises(ValueError, match="unique"):
        load_effective_news_panel(duplicate)

    nonfinite = tmp_path / "nonfinite.parquet"
    pd.DataFrame({"ticker": ["AAA"], "date": ["2020-01-02"], "f0": [np.inf]}).to_parquet(nonfinite)
    with pytest.raises(ValueError, match="finite"):
        load_effective_news_panel(nonfinite)

    partial = tmp_path / "partial.parquet"
    pd.DataFrame({"ticker": ["AAA"], "date": ["2020-01-02"], "f0": [1.0], "f1": [np.nan]}).to_parquet(partial)
    with pytest.raises(ValueError, match="finite"):
        load_effective_news_panel(partial)

    null_date = tmp_path / "null-date.parquet"
    pd.DataFrame({"ticker": ["AAA"], "date": [None], "f0": [1.0]}).to_parquet(null_date)
    with pytest.raises(ValueError, match="date"):
        load_effective_news_panel(null_date)

    offset = tmp_path / "offset.parquet"
    pd.DataFrame({"ticker": ["AAA"], "date": ["2020-01-02T23:00:00+07:00"], "f0": [1.0]}).to_parquet(offset)
    assert "2020-01-02" in {date for _ticker, date in load_effective_news_panel(offset).keys()}

    panel = load_effective_news_panel(_write_news_panel(tmp_path, [("AAA", "2020-01-02", [2.0, 1.0])]))
    with pytest.raises(ValueError, match="feature_cols"):
        attach_news([_sample_with_dates("AAA", ["2020-01-02"])], panel, ["f1", "f0"])


def test_news_attachment_rejects_independent_key_coverage_mismatch() -> None:
    panel = NewsPanel(
        values={("AAA", "2020-01-02"): np.array([1.0, 2.0])},
        feature_cols=("f0", "f1"),
        provenance={},
        eligible_keys=frozenset({("AAA", "2020-01-01")}),
    )
    with pytest.raises(RuntimeError, match="coverage"):
        attach_news([_sample_with_dates("AAA", ["2020-01-01"])], panel, ["f0", "f1"])


def test_manifest_content_hash_changes_for_targets_masks_and_tensors() -> None:
    def manifest(sample: PooledSample) -> PooledManifest:
        samples = {split: (sample,) for split in ("train", "val", "test")}
        return PooledManifest(samples=samples, exclusions={}, ticker_to_id={"AAA": 0}, preprocessing_hash="pre")

    first = manifest(_sample_with_dates("AAA", ["2020-01-02"], y=1.0))
    changed_target = manifest(_sample_with_dates("AAA", ["2020-01-02"], y=2.0))
    changed_mask = manifest(_sample_with_dates("AAA", ["2020-01-02"], mask=[1]))
    changed_price = PooledSample(
        key=SampleKey(0, "AAA", "2020-01-10"), x_price_raw=np.array([[2.0]]),
        x_news=np.zeros((1, 0)), news_mask=np.zeros(1, dtype=np.int8), y_raw=1.0,
        input_dates=("2020-01-02",),
    )

    assert first.content_hash("train") != changed_target.content_hash("train")
    assert first.content_hash("train") != changed_mask.content_hash("train")
    assert first.content_hash("train") != manifest(changed_price).content_hash("train")
    assert first.to_dict()["hashes"]["eligibility_raw_targets"] != (
        changed_target.to_dict()["hashes"]["eligibility_raw_targets"]
    )

    changed_model = _sample_with_dates("AAA", ["2020-01-02"], y=1.0)
    changed_model = PooledSample(changed_model.key, changed_model.x_price_raw, changed_model.x_news,
                                 changed_model.news_mask, changed_model.y_raw, 9.0, 1.0,
                                 changed_model.input_dates)
    changed_eval = PooledSample(changed_model.key, changed_model.x_price_raw, changed_model.x_news,
                                changed_model.news_mask, changed_model.y_raw, 9.0, 8.0,
                                changed_model.input_dates)
    assert first.content_hash("train") != manifest(changed_model).content_hash("train")
    assert manifest(changed_model).content_hash("train") != manifest(changed_eval).content_hash("train")


def test_news_provenance_fit_period_cannot_exceed_train_cutoff(tmp_path: Path) -> None:
    panel_path = _write_news_panel(tmp_path, [("AAA", "2020-01-02", [1.0, 2.0])])
    panel_path.with_suffix(".provenance.json").write_text(
        '{"fit_period_end": "2020-01-03"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fit period"):
        load_effective_news_panel(panel_path, eligible_train_cutoff="2020-01-02")

    panel_path.with_suffix(".provenance.json").unlink()
    with pytest.raises(ValueError, match="provenance"):
        load_effective_news_panel(panel_path, eligible_train_cutoff="2020-01-02")

    with pytest.raises(ValueError, match="provenance"):
        load_effective_news_panel(panel_path, require_provenance=True)


@pytest.mark.smoke
def test_real_panel_slice_and_source_timestamp_use_effective_trading_date() -> None:
    panel_path = _ROOT / "data" / "features" / "dual_group_news_panel.parquet"
    source_root = Path(os.environ.get("CRAWL_DATA_ROOT", str(_ROOT.parents[2] / "crawl_data")))
    source_path = source_root / "data" / "news_articles.csv"
    assert panel_path.exists()
    assert source_path.exists()
    with pytest.raises(ValueError, match="finite"):
        load_effective_news_panel(panel_path, tickers=["ACB"])
    raw_panel = pd.read_parquet(panel_path, filters=[("ticker", "in", ["ACB"])])
    panel_dates = set(raw_panel["date"].astype(str).str[:10])
    source = pd.read_csv(source_path, usecols=["title", "pub_date"], nrows=5000)
    record = source[source["title"].fillna("").str.contains(r"\bACB\b", regex=True)].iloc[0]
    calendar = pd.read_csv(_ROOT / "data" / "processed" / "ACB_processed.csv", usecols=["date"])["date"]
    from vendor_data_eda.phase04_news_helpers import effective_trading_date  # noqa: PLC0415

    expected = effective_trading_date(pd.Series([record["pub_date"]]), calendar).iloc[0].strftime("%Y-%m-%d")
    assert expected in panel_dates
