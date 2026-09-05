"""Package the enriched DATA needed to train VolGA / edge-fix on Google Colab (A100).

Code is pulled on Colab via ``git clone`` (see ``notebooks/train_volga_colab_a100.ipynb``), so the bundle
is DATA-ONLY: it contains ``data/processed_enriched/<market>/`` with repo-relative paths preserved, and
the notebook unpacks only that dir into the cloned ``/content/repo``. Keeping code out of the bundle avoids
shipping a stale copy alongside the git-cloned (current) code.

Run: .venv_gpu_encode/Scripts/python.exe scripts/colab/make_colab_bundle.py --market sp500_clean
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _add_dir(zf: zipfile.ZipFile, repo: Path, rel: str, exclude_suffixes=(".pyc",)) -> int:
    """Add every file under ``repo/rel`` to the zip with repo-relative POSIX names; skip dirs,
    ``__pycache__`` and the given suffixes. Returns the number of files written."""
    base = repo / rel
    n = 0
    for p in base.rglob("*"):
        if p.is_dir() or "__pycache__" in p.parts:
            continue
        if p.suffix in exclude_suffixes:
            continue
        zf.write(p, p.relative_to(repo).as_posix())
        n += 1
    return n


def build_bundle(repo: Path, market: str, out: Path) -> int:
    """Write a DATA-ONLY zip of ``data/processed_enriched/<market>/`` (repo-relative paths). Returns
    the file count. Raises ``SystemExit`` if the enriched data dir is missing."""
    data_rel = f"data/processed_enriched/{market}"
    if not (repo / data_rel).exists():
        raise SystemExit(f"missing enriched data: {data_rel} (run the clean/enrich first)")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        total = _add_dir(zf, repo, data_rel, exclude_suffixes=())   # keep all data files
    return total


def main():  # pragma: no cover - entry driver; build_bundle/_add_dir are unit-tested
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="sp500_clean")
    a = ap.parse_args()
    out = REPO / f"colab_bundle_{a.market}.zip"
    total = build_bundle(REPO, a.market, out)
    size_mb = out.stat().st_size / 1e6
    print(f"[bundle] wrote {out}  ({total} data files, {size_mb:.1f} MB) -- DATA ONLY; code comes from git clone")
    print(f"[bundle] upload this to your Google Drive; the notebook unpacks it and trains --market {a.market}")


if __name__ == "__main__":  # pragma: no cover
    main()
