"""Unit tests for the DATA-ONLY Colab bundler."""
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_colab_bundle as mb  # noqa: E402


def test_add_dir_skips_dirs_pycache_and_excluded_suffixes(tmp_path):
    repo = tmp_path
    base = repo / "d"
    (base / "sub").mkdir(parents=True)
    (base / "sub" / "keep.csv").write_text("a")          # nested file -> kept
    (base / "__pycache__").mkdir()
    (base / "__pycache__" / "x.pyc").write_bytes(b"x")   # __pycache__ -> skipped
    (base / "drop.pyc").write_bytes(b"x")                # excluded suffix -> skipped
    (base / "keep.txt").write_text("t")
    out = repo / "b.zip"
    with zipfile.ZipFile(out, "w") as zf:
        n = mb._add_dir(zf, repo, "d", exclude_suffixes=(".pyc",))
    names = zipfile.ZipFile(out).namelist()
    assert "d/sub/keep.csv" in names and "d/keep.txt" in names
    assert not any(x.endswith(".pyc") for x in names)
    assert n == 2


def test_build_bundle_is_data_only(tmp_path):
    repo = tmp_path
    data = repo / "data" / "processed_enriched" / "mkt"
    data.mkdir(parents=True)
    (data / "AAA.csv").write_text("date,close\n2020-01-01,1\n")
    (data / "BBB.csv").write_text("date,close\n2020-01-01,2\n")
    # a code dir that must NOT end up in the bundle
    (repo / "baselines" / "x" / "code").mkdir(parents=True)
    (repo / "baselines" / "x" / "code" / "run.py").write_text("print(1)\n")
    out = repo / "bundle.zip"
    n = mb.build_bundle(repo, "mkt", out)
    names = zipfile.ZipFile(out).namelist()
    assert n == 2
    assert set(names) == {"data/processed_enriched/mkt/AAA.csv", "data/processed_enriched/mkt/BBB.csv"}
    assert not any(x.endswith(".py") for x in names)


def test_build_bundle_missing_data_raises(tmp_path):
    with pytest.raises(SystemExit):
        mb.build_bundle(tmp_path, "nope", tmp_path / "b.zip")
