"""Integration test for aggregate_news_sources.main(): tiny CSV fixtures -> unified output.

Covers _read (on_bad_lines counting), normalize_a/b, dedup, main() orchestration.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]  # project root
sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data_aggregation import aggregate_news_sources as agg


def _write_family_a(raw: Path):
    """Family A (title-only): data.csv schema."""
    pd.DataFrame({
        "id": ["1", "2"],
        "title": ["VCB tin 1", "TCB tin 2"],
        "source": ["KBSV", "MBS"],
        "date": ["01/01/2024", "02/01/2024"],   # DD/MM/YYYY
        "pdf_url": ["http://a", "http://b"],
        "pdf_filename": ["a.pdf", "b.pdf"],
        "downloaded_at": ["", ""],
    }).to_csv(raw / "data.csv", index=False)


def _write_family_b(raw: Path):
    """Family B (lead/category): cafef schema."""
    pd.DataFrame({
        "id": ["10"],
        "source": ["cafef"],
        "title": ["VCB cafef"],
        "category": ["mkt"],
        "pub_date": ["2024-01-03T08:00:00+0700"],   # ISO
        "article_url": ["http://c"],
        "author": [""],
        "lead": ["lead c"],
        "collected_at": [""],
    }).to_csv(raw / "cafef_articles.csv", index=False)


def test_main_aggregates_two_families(tmp_path, monkeypatch):
    raw = tmp_path / "data"
    raw.mkdir()
    out = tmp_path / "aggregated"
    _write_family_a(raw)
    _write_family_b(raw)

    monkeypatch.setattr(agg, "RAW_DIR", raw)
    monkeypatch.setattr(agg, "OUT_DIR", out)
    agg.main()

    df = pd.read_csv(out / "unified_articles.csv", dtype=str)
    assert len(df) == 3  # 2 (Family A) + 1 (Family B), all unique
    assert set(df["source"]) >= {"KBSV", "MBS", "cafef"}
    # date normalized to YYYY-MM-DD for both formats
    dates = set(df["date"].dropna())
    assert "2024-01-01" in dates or "2024-01-02" in dates  # Family A parsed
    assert "2024-01-03" in dates  # Family B ISO parsed
    # Family B lead carried over
    cafef = df[df["source"] == "cafef"].iloc[0]
    assert cafef["lead"] == "lead c"


def test_main_dedups_same_title(tmp_path, monkeypatch):
    """Two files with SAME title (no url) -> deduped to 1 (dedup is by url/title-hash)."""
    raw = tmp_path / "data"
    raw.mkdir()
    out = tmp_path / "aggregated"
    # two files, same title, empty pdf_url -> same title-hash -> deduped
    pd.DataFrame({"id": ["1"], "title": ["same title"], "source": ["s1"], "date": ["01/01/2024"],
                  "pdf_url": [""], "pdf_filename": [""], "downloaded_at": [""]}).to_csv(raw / "data.csv", index=False)
    pd.DataFrame({"id": ["2"], "title": ["same title"], "source": ["s2"], "date": ["02/01/2024"],
                  "pdf_url": [""], "pdf_filename": [""], "downloaded_at": [""]}).to_csv(raw / "data_2021_2025.csv", index=False)

    monkeypatch.setattr(agg, "RAW_DIR", raw)
    monkeypatch.setattr(agg, "OUT_DIR", out)
    agg.main()

    df = pd.read_csv(out / "unified_articles.csv", dtype=str)
    assert len(df) == 1  # deduped by title-hash


def test_read_counts_malformed_rows(tmp_path, capsys):
    """[MED-5] _read counts + logs malformed rows (extra unquoted comma)."""
    raw = tmp_path / "data"
    raw.mkdir()
    # header (8 cols) + 1 good row + 1 malformed row (extra comma -> 9 fields)
    (raw / "cafef_articles.csv").write_text(
        "id,source,title,category,pub_date,article_url,author,lead\n"
        "1,cafef,good,mkt,2024-01-01T08:00:00+0700,http://a,,gl\n"
        "2,cafef,bad,extra,comma,here,2024-01-02,http://b,,bl\n",
        encoding="utf-8",
    )
    orig = agg.RAW_DIR
    agg.RAW_DIR = raw
    try:
        df = agg._read("cafef_articles.csv")
        captured = capsys.readouterr()
    finally:
        agg.RAW_DIR = orig
    assert len(df) >= 1  # good row kept
    assert "dropped" in captured.out  # [MED-5] malformed drop logged


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
