"""Integration test for extract_pilot_body: mock fitz + tiny unified CSV -> body column.

Covers extract_body() (fitz open/auth/get_text) and main() (sample, join, write).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.body_pilot import extract_pilot_body as ep


class _FakePage:
    def __init__(self, text):
        self._t = text

    def get_text(self, mode):
        return self._t


class _FakeDoc:
    needs_pass = False

    def __init__(self, text):
        self._pages = [_FakePage(text)]

    def __iter__(self):
        return iter(self._pages)

    def close(self):
        pass


def test_extract_body_returns_text(monkeypatch):
    monkeypatch.setattr(ep.fitz, "open", lambda p: _FakeDoc("hello world body"))
    assert ep.extract_body(Path("fake.pdf")) == "hello world body"


def test_extract_body_empty_on_open_failure(monkeypatch):
    def _boom(_p):
        raise RuntimeError("corrupt")

    monkeypatch.setattr(ep.fitz, "open", _boom)
    assert ep.extract_body(Path("bad.pdf")) == ""


def test_main_joins_body_to_unified(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdf"
    pdf_dir.mkdir()
    (pdf_dir / "report.pdf").write_bytes(b"%PDF-1.4 dummy")
    unified_csv = tmp_path / "unified.csv"
    pd.DataFrame({
        "unified_id": ["1", "2"],
        "pdf_filename": ["report.pdf", ""],
        "title": ["has body", "no body"],
        "lead": ["", ""],
        "date": ["2024-01-01", "2024-01-02"],
        "source": ["s1", "s2"],
    }).to_csv(unified_csv, index=False)
    out_csv = tmp_path / "out.csv"

    monkeypatch.setattr(ep.fitz, "open", lambda p: _FakeDoc("extracted body content"))
    monkeypatch.setattr(ep, "PDF_DIR", pdf_dir)
    monkeypatch.setattr(ep, "UNIFIED", unified_csv)
    monkeypatch.setattr(ep, "OUT_CSV", out_csv)
    monkeypatch.setattr("sys.argv", ["ep", "--n", "10"])
    ep.main()

    df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
    assert "body" in df.columns
    matched = df[df["body_source"] == "pdf_pilot"]
    assert len(matched) == 1
    assert matched.iloc[0]["body"] == "extracted body content"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
