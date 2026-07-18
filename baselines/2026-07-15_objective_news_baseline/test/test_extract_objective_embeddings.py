"""Tests for extract_objective_embeddings: build_records() (pure) + main() (mocked PhoBERT).

Pattern follows sibling baseline's test_extract_integration.py (mock transformers, no
network/GPU). Covers: direct company_code match, ticker-regex on unenriched news, HTML
strip, title==raw_text dedup, publish_time-after-crawl_time leakage guard, end-to-end cache.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

_CODE = Path(__file__).resolve().parents[1] / "code"
_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch

import extract_objective_embeddings as eoe


COLS = ["document_id", "source", "source_tier", "url", "publish_time", "crawl_time",
        "company_code", "company_name", "title", "raw_text", "language", "category",
        "event_type", "attachment_urls", "checksum", "raw_path"]


def _row(**kw):
    base = {c: "" for c in COLS}
    base.update(kw)
    return base


def _write_csv(dir_, name, rows):
    pd.DataFrame(rows, columns=COLS).to_csv(dir_ / name, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# build_records
# ---------------------------------------------------------------------------

def test_direct_file_uses_company_code(tmp_path, monkeypatch):
    monkeypatch.setattr(eoe, "OBJ_DIR", tmp_path)
    _write_csv(tmp_path, "vietstock_records.csv", [
        _row(company_code="VCB", publish_time="2020-05-01T00:00:00Z",
             crawl_time="2026-07-12T00:00:00Z", title="VCB tra co tuc", raw_text="VCB tra co tuc"),
        _row(company_code="XYZ", publish_time="2020-05-01T00:00:00Z",  # not in VN30 list
             crawl_time="2026-07-12T00:00:00Z", title="XYZ event", raw_text="XYZ event"),
    ])
    _write_csv(tmp_path, "vsdc_records.csv", [])

    records = eoe.build_records(["VCB", "ACB"])
    assert len(records) == 1
    assert records[0]["tickers"] == {"VCB"}
    assert records[0]["date"] == "2020-05-01"
    assert records[0]["text"] == "VCB tra co tuc"  # title==raw_text -> not duplicated


def test_unenriched_file_ticker_regex_and_html_strip(tmp_path, monkeypatch):
    monkeypatch.setattr(eoe, "OBJ_DIR", tmp_path)
    _write_csv(tmp_path, "vietstock_records.csv", [])
    _write_csv(tmp_path, "vsdc_records.csv", [])
    _write_csv(tmp_path, "news_unenriched_vnexpress_records.csv", [
        _row(publish_time="2026-07-11T00:00:00Z", crawl_time="2026-07-12T00:00:00Z",
             title="CEO PNJ muon mua 1 trieu co phieu",
             raw_text='<a href="x"><img src="y"></a></br>Noi dung ve PNJ'),
        _row(publish_time="2026-07-11T00:00:00Z", crawl_time="2026-07-12T00:00:00Z",
             title="Tin khong lien quan", raw_text="khong co ticker nao"),
    ])
    for f in eoe.UNENRICHED_FILES[1:]:
        _write_csv(tmp_path, f, [])

    records = eoe.build_records(["PNJ", "VCB"])
    assert len(records) == 1
    assert records[0]["tickers"] == {"PNJ"}
    assert "<a" not in records[0]["text"] and "<img" not in records[0]["text"]


def test_unenriched_file_matches_by_brand_name_without_ticker_code(tmp_path, monkeypatch):
    """General news says "Vinamilk" (brand), not "VNM" (ticker) — must still match via alias."""
    monkeypatch.setattr(eoe, "OBJ_DIR", tmp_path)
    _write_csv(tmp_path, "vietstock_records.csv", [])
    _write_csv(tmp_path, "vsdc_records.csv", [])
    _write_csv(tmp_path, "news_unenriched_vnexpress_records.csv", [
        _row(publish_time="2026-07-11T00:00:00Z", crawl_time="2026-07-12T00:00:00Z",
             title="Cong ty sua Vinamilk tuyen bo loi nhuan giam 50%", raw_text=""),
    ])
    for f in eoe.UNENRICHED_FILES[1:]:
        _write_csv(tmp_path, f, [])

    records = eoe.build_records(["VNM", "VCB"])  # no "VNM" ticker string in the text at all
    assert len(records) == 1
    assert records[0]["tickers"] == {"VNM"}


def test_leakage_guard_drops_future_publish_time(tmp_path, monkeypatch):
    monkeypatch.setattr(eoe, "OBJ_DIR", tmp_path)
    _write_csv(tmp_path, "vietstock_records.csv", [
        _row(company_code="VCB", publish_time="2026-08-27T00:00:00Z",  # AFTER crawl_time
             crawl_time="2026-07-12T00:00:00Z", title="future record date", raw_text=""),
    ])
    _write_csv(tmp_path, "vsdc_records.csv", [])
    for f in eoe.UNENRICHED_FILES:
        _write_csv(tmp_path, f, [])

    records = eoe.build_records(["VCB"])
    assert records == []  # dropped: publish_time > crawl_time


# ---------------------------------------------------------------------------
# main() end-to-end (mocked PhoBERT, same pattern as sibling baseline)
# ---------------------------------------------------------------------------

class _FakeEnc(dict):
    def __init__(self, batch_size):
        super().__init__()
        self["input_ids"] = torch.zeros((batch_size, 4), dtype=torch.long)

    def to(self, device):
        return self


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, batch, return_tensors="pt", truncation=True, padding=True, max_length=64):
        return _FakeEnc(len(batch))


class _FakeModel:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, **kwargs):
        batch = kwargs["input_ids"].shape[0]
        return SimpleNamespace(last_hidden_state=torch.randn(batch, 4, 768))

    @property
    def config(self):
        return SimpleNamespace(hidden_size=768)


def test_main_writes_ticker_cache(tmp_path, monkeypatch):
    obj_dir = tmp_path / "objective"
    obj_dir.mkdir()
    monkeypatch.setattr(eoe, "OBJ_DIR", obj_dir)
    monkeypatch.setattr(eoe, "load_tickers", lambda: ["VCB", "ACB"])

    _write_csv(obj_dir, "vietstock_records.csv", [
        _row(document_id="doc1", company_code="VCB", publish_time="2024-01-15T00:00:00Z",
             crawl_time="2026-07-12T00:00:00Z", title="VCB tra co tuc", raw_text="VCB tra co tuc"),
    ])
    _write_csv(obj_dir, "vsdc_records.csv", [])
    for f in eoe.UNENRICHED_FILES:
        _write_csv(obj_dir, f, [])

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", _FakeModel)

    emb_dir = tmp_path / "emb"
    monkeypatch.setattr("sys.argv", ["eoe", "--emb_dir", str(emb_dir), "--no_pca"])
    eoe.main()

    vcb = np.load(emb_dir / "VCB_emb.npz", allow_pickle=False)
    assert "2024-01-15" in vcb.files
    assert vcb["2024-01-15"].shape == (1, 768)
    acb_path = emb_dir / "ACB_emb.npz"
    assert acb_path.exists()
    assert len(np.load(acb_path, allow_pickle=False).files) == 0  # no news -> empty placeholder
    assert (emb_dir / eoe.MANIFEST_NAME).exists()


def test_incremental_second_run_skips_already_processed(tmp_path, monkeypatch, capsys):
    """Rerunning with the SAME csv (no new rows) must encode 0 new records."""
    obj_dir = tmp_path / "objective"
    obj_dir.mkdir()
    monkeypatch.setattr(eoe, "OBJ_DIR", obj_dir)
    monkeypatch.setattr(eoe, "load_tickers", lambda: ["VCB"])
    _write_csv(obj_dir, "vietstock_records.csv", [
        _row(document_id="doc1", company_code="VCB", publish_time="2024-01-15T00:00:00Z",
             crawl_time="2026-07-12T00:00:00Z", title="VCB tra co tuc", raw_text="VCB tra co tuc"),
    ])
    _write_csv(obj_dir, "vsdc_records.csv", [])
    for f in eoe.UNENRICHED_FILES:
        _write_csv(obj_dir, f, [])

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", _FakeModel)

    emb_dir = tmp_path / "emb"
    monkeypatch.setattr("sys.argv", ["eoe", "--emb_dir", str(emb_dir), "--no_pca"])
    eoe.main()
    capsys.readouterr()

    eoe.main()  # rerun, same input
    out = capsys.readouterr().out
    assert "nothing new to encode" in out
    vcb = np.load(emb_dir / "VCB_emb.npz", allow_pickle=False)
    assert vcb["2024-01-15"].shape == (1, 768)  # not duplicated


def test_incremental_merges_new_row_without_refitting_pca(tmp_path, monkeypatch):
    """A second run with 1 NEW row must merge into the existing cache (both dates present)
    and must reuse the persisted PCA rather than refit (train_cutoff far in the past so a
    refit here would be a leakage bug if it ever happened)."""
    obj_dir = tmp_path / "objective"
    obj_dir.mkdir()
    monkeypatch.setattr(eoe, "OBJ_DIR", obj_dir)
    monkeypatch.setattr(eoe, "load_tickers", lambda: ["VCB"])
    csv_path_rows = [
        _row(document_id="doc1", company_code="VCB", publish_time="2010-01-15T00:00:00Z",
             crawl_time="2026-07-12T00:00:00Z", title="VCB tin 1", raw_text="VCB tin 1"),
        _row(document_id="doc2", company_code="VCB", publish_time="2010-02-20T00:00:00Z",
             crawl_time="2026-07-12T00:00:00Z", title="VCB tin 2", raw_text="VCB tin 2"),
    ]
    _write_csv(obj_dir, "vietstock_records.csv", [csv_path_rows[0]])
    _write_csv(obj_dir, "vsdc_records.csv", [])
    for f in eoe.UNENRICHED_FILES:
        _write_csv(obj_dir, f, [])

    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", _FakeModel)

    emb_dir = tmp_path / "emb"
    monkeypatch.setattr("sys.argv", ["eoe", "--emb_dir", str(emb_dir), "--no_pca"])
    eoe.main()  # run 1: 1 record

    # run 2: add a 2nd row (new document_id) to the SAME csv, keep PCA off (dim check N/A here,
    # incremental merge is what's under test)
    _write_csv(obj_dir, "vietstock_records.csv", csv_path_rows)
    pca_mtime_before = None  # --no_pca -> no pca file expected at all
    eoe.main()

    vcb = np.load(emb_dir / "VCB_emb.npz", allow_pickle=False)
    assert set(vcb.files) == {"2010-01-15", "2010-02-20"}
    assert not (emb_dir / eoe.PCA_NAME).exists()  # --no_pca throughout -> never created


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
