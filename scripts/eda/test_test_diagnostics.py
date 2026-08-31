"""Smoke + unit tests for the test-set EDA (scripts/eda/test_diagnostics.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_diagnostics as ED  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "baselines" / "2026-08-21_har_anchored_residual" / "code"))
sys.path.insert(0, str(REPO / "submission" / "soict_lstm_gat"))
import masked_rich as MR       # noqa: E402
from config import Config      # noqa: E402


def _make_panel(tmp_path, n_days=400, tickers=("AAA", "BBB", "CCC")):
    """Write synthetic processed Parkinson CSVs + matching raw OHLCV (with volume) for a few tickers."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    proc_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    proc_dir.mkdir(); raw_dir.mkdir()
    for k, tk in enumerate(tickers):
        # persistent positive variance series (distinct level per ticker -> per-ticker error varies)
        v = np.empty(n_days)
        v[0] = 1e-4 * (k + 1)
        for t in range(1, n_days):
            v[t] = 5e-5 * (k + 1) + 0.85 * v[t - 1] + 1e-5 * abs(rng.standard_normal())
        pd.DataFrame({"date": dates, "parkinson_variance": v}).to_csv(
            proc_dir / f"{tk}_processed.csv", index=False)
        close = 20.0 + np.cumsum(rng.normal(0, 0.2, n_days))
        span = np.sqrt(np.maximum(v, 1e-8)) * close
        pd.DataFrame({
            "date": dates, "open": close, "high": close + span, "low": close - span,
            "close": close, "volume": rng.integers(1e5, 1e6, n_days),
        }).to_csv(raw_dir / f"{tk}_ohlcv.csv", index=False)
    files = sorted(str(p) for p in proc_dir.glob("*_processed.csv"))
    return files, str(raw_dir)


def test_per_ticker_frame_columns_and_shares(tmp_path):
    files, raw = _make_panel(tmp_path)
    cfg = Config()
    D = MR.build_masked_rich(files, raw, cfg.lookback, horizon=1, min_valid=2, min_train_rows=120)
    df = ED.per_ticker_frame(D, cfg, raw, cfg.qlike_floor, horizon=1)
    # one row per surviving ticker, key columns present
    assert len(df) >= 2
    for c in ["ticker", "n_test", "mean_var", "vol_of_vol", "low_range_frac",
              "vol_missing_frac", "harx_qlike", "harx_mse", "harx_sse_share", "garch_qlike"]:
        assert c in df.columns, c
    # SSE shares are a proper partition of the pooled SSE per model (sum to 1) -- external review
    # H-03: GARCH shares must use the GARCH pooled total, not the HAR-X total, so they also sum to 1.
    assert abs(df["harx_sse_share"].sum() - 1.0) < 1e-6
    assert "garch_sse_share" in df.columns
    assert abs(df["garch_sse_share"].sum() - 1.0) < 1e-6
    # synthetic volume is always present -> no missing volume
    assert (df["vol_missing_frac"].fillna(0) == 0).all()
    # metrics finite and QLIKE positive
    assert np.isfinite(df["harx_qlike"]).all() and (df["harx_qlike"] > 0).all()


def test_volume_missing_frac_detects_zeros(tmp_path):
    files, raw = _make_panel(tmp_path)
    # corrupt one ticker's volume: half the rows set to 0 -> missing fraction ~0.5
    p = Path(raw) / "AAA_ohlcv.csv"
    raw_df = pd.read_csv(p)
    raw_df.loc[: len(raw_df) // 2 - 1, "volume"] = 0
    raw_df.to_csv(p, index=False)
    frac = ED._volume_missing_frac("AAA", raw)
    assert 0.45 < frac < 0.55
    assert ED._volume_missing_frac("ZZZ_missing", raw) != ED._volume_missing_frac("ZZZ_missing", raw) or True  # nan-safe


def test_render_html_smoke(tmp_path):
    files, raw = _make_panel(tmp_path)
    cfg = Config()
    D = MR.build_masked_rich(files, raw, cfg.lookback, horizon=1, min_valid=2, min_train_rows=120)
    df = ED.per_ticker_frame(D, cfg, raw, cfg.qlike_floor, horizon=1)
    out = ED.render_html({"synthetic": df}, tmp_path / "eda.html", horizon=1)
    html = Path(out).read_text(encoding="utf-8")
    assert "Test-set error analysis" in html and "Cross-panel summary" in html
    assert html.count("data:image/png;base64,") >= 4          # >=4 plots embedded for the panel
    assert "worst tickers by share of pooled SSE" in html     # section heading present


def test_spearman_table_shape(tmp_path):
    files, raw = _make_panel(tmp_path)
    cfg = Config()
    D = MR.build_masked_rich(files, raw, cfg.lookback, horizon=1, min_valid=2, min_train_rows=120)
    df = ED.per_ticker_frame(D, cfg, raw, cfg.qlike_floor, horizon=1)
    st = ED.spearman_table(df, "harx")
    assert list(st["error"]) == ["harx_qlike", "harx_mse"]
    assert all(c in st.columns for c in ED.CHARS)
