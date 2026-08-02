"""Offline: extract PhoBERT (frozen) embeddings for ALL articles (no ticker filter) + PCA -> market_emb.npz.

Market-level fallback: ~80% of articles are macro/market-wide (don't match a VN30 ticker).
This script embeds ALL of them, grouped by DATE, so every trading day gets a market-news
representation (dense, ~100% coverage) to fall back on when a stock has no ticker-specific news.

ISOLATED: reads unified_articles.csv (READ ONLY). Writes data/sentiment_embedding/market_emb.npz.
PCA fit on TRAIN-period articles only (date < 2020-01-01) — no leakage (HIGH-1 lesson).

ENV: requires transformers<5 (4.57.6 works). pip install "transformers<5" sentencepiece

Run:
  python baselines/2026-07-08_market_fallback/code/extract_market_embeddings.py
  python baselines/2026-07-08_market_fallback/code/extract_market_embeddings.py --no_pca --dim 768
"""
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict

_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

UNIFIED = _ROOT.parent / "crawl_data" / "aggregated" / "unified_articles.csv"  # [HIGH-2]
OUT_DIR = _ROOT / "data" / "sentiment_embedding"
OUT_FILE = OUT_DIR / "market_emb.npz"

TRAIN_CUTOFF = "2020-01-01"  # PCA fit only on articles before this (train split approximation)


def _norm_date(s):
    s = str(s).strip()
    for sep in (' ', 'T'):
        if sep in s:
            s = s.split(sep)[0]
    return s[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--use_pca", dest="use_pca", action="store_true", default=True)
    ap.add_argument("--no_pca", dest="use_pca", action="store_false")
    ap.add_argument("--train_cutoff", default=TRAIN_CUTOFF,
                    help="PCA fit on articles before this date (approx of train split)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--input", default=str(UNIFIED), help="Input articles CSV")
    ap.add_argument("--emb_dir", default=str(OUT_DIR), help="Output cache dir")
    ap.add_argument("--use_body", action="store_true", help="Use title+body+lead (needs 'body' column)")
    args = ap.parse_args()

    out_dir = Path(args.emb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "market_emb.npz"
    print(f"[market] loading {Path(args.input).name} use_body={args.use_body}")
    df = pd.read_csv(args.input, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    has_body = "body" in df.columns

    # ALL articles with a parseable date + non-empty text (NO ticker filter)
    DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    records = []
    skipped_date = 0
    for r in df.itertuples(index=False):
        d = _norm_date(getattr(r, "date", "") or "")
        if not DATE_RE.match(d):
            skipped_date += 1
            continue
        title = getattr(r, "title", "") or ""
        lead = getattr(r, "lead", "") or ""
        body = (getattr(r, "body", "") or "") if (args.use_body and has_body) else ""
        text = (title + " " + body + " " + lead).strip() if body else (title + " " + lead).strip()
        if not text:
            continue
        text = text[: args.max_len * 6]
        records.append({"date": d, "text": text})
    if skipped_date:
        print(f"[market][warn] skipped {skipped_date} rows with non-YYYY-MM-DD date")
    print(f"[market] ALL articles (no ticker filter): {len(records)}")
    if not records:
        print("[market] nothing to do.")
        return

    # --- PhoBERT encode (frozen) ---
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"[market] loading {args.model} (requires transformers<5)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = [r["text"] for r in records]
    raw_dim = model.config.hidden_size
    embs = np.zeros((len(records), raw_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True,
                      padding=True, max_length=args.max_len).to(device)
            cls = model(**enc).last_hidden_state[:, 0, :]
            embs[i:i + len(batch)] = cls.cpu().numpy()
            if (i // args.batch_size) % 50 == 0:
                print(f"  encoded {i+len(batch)}/{len(texts)}")

    # [MEDIUM-6 lesson] fail loud on non-finite
    assert np.isfinite(embs).all(), \
        f"PhoBERT produced non-finite embeddings ({int(np.isnan(embs).sum())} NaN)"
    print(f"[market] raw embeddings: {embs.shape}")

    # --- PCA (fit on train-period articles ONLY, no widen — HIGH-1 lesson) ---
    if args.use_pca and args.dim < raw_dim:
        from sklearn.decomposition import PCA
        train_mask = np.array([r["date"] < args.train_cutoff for r in records])
        n_train = int(train_mask.sum())
        if n_train < 2:   # [MED-8] can't fit PCA on <2 samples -> degrade to raw
            print(f"[market][warn] only {n_train} train articles — skipping PCA (raw {raw_dim}-d)")
        else:
            if n_train < args.dim:
                args.dim = max(1, n_train - 1)
                print(f"[market][warn] train articles ({n_train}) < dim; REDUCED dim to {args.dim}")
            pca = PCA(n_components=args.dim).fit(embs[train_mask])
            embs = pca.transform(embs).astype(np.float32)
            print(f"[market] PCA {raw_dim}->{args.dim} (fit on {n_train} train articles, "
                  f"explained var: {pca.explained_variance_ratio_.sum():.3f})")

    # --- Group by date -> market_emb.npz ---
    by_date = defaultdict(list)
    for i, r in enumerate(records):
        by_date[r["date"]].append(embs[i])
    save = {d: np.stack(v, axis=0) for d, v in by_date.items()}
    np.savez_compressed(OUT_FILE, **save)
    n_days = len(save)
    total = sum(len(v) for v in save.values())
    print(f"[market] wrote {OUT_FILE}: {n_days} dates, {total} article-vectors (dim={embs.shape[1]})")


if __name__ == "__main__":
    main()
