"""Pilot: sample N PDFs from crawl_data/data/pdf/, extract body (PyMuPDF), join to
unified_articles.csv by pdf_filename -> unified_articles_pilot_body.csv.

Quick test whether article BODY text (vs title-only) moves volatility DirAcc.
Isolated consumer-side experiment (crawling is external). Self-contained.

Run:
  python -m src.body_pilot.extract_pilot_body --n 300
  python -m src.body_pilot.extract_pilot_body --n 300 --seed 42
"""
import argparse
import random
from pathlib import Path

import fitz  # [MED-4] top-level: fail loud if pymupdf missing (was silently returning "" for ALL PDFs)
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_CRAWL = _ROOT.parents[1] / "crawl_data"  # [HIGH-2] sibling crawl_data dir (not hardcoded absolute)

PDF_DIR = _CRAWL / "data" / "pdf"
UNIFIED = _CRAWL / "aggregated" / "unified_articles.csv"
OUT_CSV = _CRAWL / "aggregated" / "unified_articles_pilot_body.csv"


def extract_body(pdf_path: Path) -> str:
    """Extract text from a PDF via PyMuPDF. '' on encrypted/corrupt/scanned."""
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""
    if doc.needs_pass:
        try:
            if not doc.authenticate(""):
                doc.close()
                return ""
        except Exception:
            doc.close()
            return ""
    parts = []
    try:
        for page in doc:
            t = page.get_text("text") or ""
            if t.strip():
                parts.append(t)
    finally:
        doc.close()
    return "\n".join(parts).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of PDFs to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"[pilot] found {len(pdfs)} PDFs in {PDF_DIR}")
    if not pdfs:
        print("[pilot] no PDFs — nothing to do.")
        return
    random.seed(args.seed)
    sample = random.sample(pdfs, min(args.n, len(pdfs)))
    print(f"[pilot] sampled {len(sample)} PDFs (seed={args.seed})")

    # --- extract body ---
    body_map = {}  # pdf_filename (basename) -> body text
    n_empty = 0
    for i, p in enumerate(sample):
        body = extract_body(p)
        body_map[p.name] = body
        if not body:
            n_empty += 1
        if (i + 1) % 50 == 0:
            print(f"  extracted {i+1}/{len(sample)}")
    n_with_body = sum(1 for v in body_map.values() if v)
    print(f"[pilot] body extracted: {n_with_body}/{len(sample)} non-empty "
          f"({n_empty} scanned/encrypted/empty)")

    # --- join to unified_articles by pdf_filename ---
    df = pd.read_csv(UNIFIED, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    df["body"] = ""
    df["body_source"] = ""
    fnames = df["pdf_filename"].astype(str).str.strip()
    for idx in df.index:
        fn = fnames[idx]
        if fn and fn in body_map and body_map[fn]:
            df.at[idx, "body"] = body_map[fn]
            df.at[idx, "body_source"] = "pdf_pilot"
    n_matched = (df["body_source"] == "pdf_pilot").sum()
    print(f"[pilot] matched {n_matched} unified rows with pilot body")

    lens = [len(b) for b in df.loc[df["body_source"] == "pdf_pilot", "body"]]
    if lens:
        lens_sorted = sorted(lens)
        print(f"[pilot] body length: min={lens_sorted[0]} "
              f"median={lens_sorted[len(lens_sorted)//2]} max={lens_sorted[-1]}")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[pilot] wrote {OUT_CSV} ({len(df)} rows, {n_matched} with body)")


if __name__ == "__main__":
    main()
