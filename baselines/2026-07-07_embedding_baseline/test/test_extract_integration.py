"""Integration test for extract_embeddings: mock PhoBERT + tiny CSV -> cache.

Covers main() records-build (incl. MED-7: ticker matched on full content pre-truncate),
encode loop (mocked), cache write. Mocks transformers + load_tickers (no network/GPU).
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
import torch
import pandas as pd

import extract_embeddings as ee


class _FakeEnc(dict):
    """Fake tokenizer output — dict subclass so model(**enc) unpacks; has .to(device)."""
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


def _setup_transformers_mock(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModel", _FakeModel)
    monkeypatch.setattr(ee, "load_tickers", lambda: ["VCB"])


def test_use_body_finds_ticker_only_in_body(tmp_path, monkeypatch):
    """[MED-7] ticker in BODY (not title) is matched when --use_body (full-content search)."""
    _setup_transformers_mock(monkeypatch)
    csv = tmp_path / "in.csv"
    pd.DataFrame({
        "unified_id": ["1"], "source": ["cafef"],
        "title": ["tin tai chinh"],          # NO ticker in title
        "lead": [""],
        "body": ["body text about VCB"],     # ticker only in body
        "date": ["2024-01-15"],
        "pub_datetime": [""], "url": [""], "pdf_url": [""],
        "pdf_filename": [""], "collected_at": [""], "origin_file": ["x"],
    }).to_csv(csv, index=False)
    emb_dir = tmp_path / "emb"

    monkeypatch.setattr("sys.argv", ["ee", "--input", str(csv), "--emb_dir", str(emb_dir),
                                    "--use_body", "--max_len", "64", "--no_pca"])
    ee.main()

    vcb = np.load(emb_dir / "VCB_emb.npz", allow_pickle=False)
    assert "2024-01-15" in vcb.files      # article matched via body ticker
    assert vcb["2024-01-15"].shape == (1, 768)


def test_without_use_body_drops_ticker_only_in_body(tmp_path, monkeypatch):
    """Without --use_body, ticker only in body (not title+lead) → article dropped (no match)."""
    _setup_transformers_mock(monkeypatch)
    csv = tmp_path / "in.csv"
    pd.DataFrame({
        "unified_id": ["1"], "source": ["cafef"],
        "title": ["tin tai chinh"], "lead": [""],
        "body": ["body about VCB"],
        "date": ["2024-01-15"],
        "pub_datetime": [""], "url": [""], "pdf_url": [""],
        "pdf_filename": [""], "collected_at": [""], "origin_file": ["x"],
    }).to_csv(csv, index=False)
    emb_dir = tmp_path / "emb"

    monkeypatch.setattr("sys.argv", ["ee", "--input", str(csv), "--emb_dir", str(emb_dir),
                                    "--max_len", "64", "--no_pca"])  # no --use_body
    ee.main()

    # VCB cache empty (no match without body) — file may not exist or be empty
    vcb_path = emb_dir / "VCB_emb.npz"
    if vcb_path.exists():
        assert len(np.load(vcb_path, allow_pickle=False).files) == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
