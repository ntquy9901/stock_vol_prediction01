"""Offline: extract PhoBERT (frozen) embeddings for ticker-matched news + PCA -> cache .npz.

ISOLATED: reads unified_articles.csv + data/processed (READ ONLY). Writes data/sentiment_embedding/.

PCA is fit on TRAIN-period articles only (date < 2020-01-01) to avoid leakage from val/test.

ENV GOTCHA: requires `transformers<5` (4.57.x). See memory: project-sentiment-baseline-status.
  pip install "transformers<5" sentencepiece

Run:
  python baselines/2026-07-07_embedding_baseline/code/extract_embeddings.py
  python baselines/2026-07-07_embedding_baseline/code/extract_embeddings.py --no_pca --dim 768
"""
import sys
import csv
import re
import argparse
from pathlib import Path
from collections import defaultdict

# bootstrap paths (rule 3.F.4): project root + this code/ dir
_ROOT = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

# [HIGH-2] sibling crawl_data dir (derived from project root, not hardcoded absolute)
UNIFIED = _ROOT.parents[1] / "crawl_data" / "aggregated" / "unified_articles.csv"
STOCK_DIR = _ROOT / "data" / "processed"
SUMMARY = STOCK_DIR / "processing_summary.csv"
OUT_DIR = _ROOT / "data" / "sentiment_embedding"

TRAIN_CUTOFF = "2020-01-01"  # PCA fit only on articles before this (train split)


def load_tickers():
    with open(SUMMARY, encoding="utf-8-sig") as f:
        return [r["ticker"].strip() for r in csv.DictReader(f) if (r.get("ticker") or "").strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/phobert-base",
                    help="HF encoder for embeddings (default PhoBERT, VN-specific)")
    ap.add_argument("--dim", type=int, default=64, help="PCA output dim (ignored if --no_pca)")
    ap.add_argument("--use_pca", dest="use_pca", action="store_true", default=True)
    ap.add_argument("--no_pca", dest="use_pca", action="store_false",
                    help="Keep raw 768-d (no PCA)")
    ap.add_argument("--train_cutoff", default=TRAIN_CUTOFF,
                    help="PCA fit on articles before this date. NOTE: the real train/val/test "
                         "split is by row-index (0.7), not calendar — this cutoff is a best-effort "
                         "approximation. For strict correctness, fit PCA inside the dataset post-split.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64, help="tokenize max_length (titles are short)")
    ap.add_argument("--input", default=str(UNIFIED),
                    help="Input articles CSV (default unified_articles.csv)")
    ap.add_argument("--emb_dir", default=str(OUT_DIR),
                    help="Output embedding cache dir (default data/sentiment_embedding)")
    ap.add_argument("--use_body", action="store_true",
                    help="Use title+body text (needs 'body' column in --input). Default: title+lead.")
    args = ap.parse_args()

    out_dir = Path(args.emb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = load_tickers()
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in tickers) + r")\b", re.IGNORECASE)
    print(f"[extract] VN30 tickers: {len(tickers)}")

    df = pd.read_csv(args.input, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    has_body_col = "body" in df.columns

    # Build per-article records for articles matching >=1 VN30 ticker
    records = []
    for r in df.itertuples(index=False):
        d = (getattr(r, "date", "") or "").strip()
        if len(d) < 10:
            continue
        title = getattr(r, "title", "") or ""
        lead = getattr(r, "lead", "") or ""
        body = (getattr(r, "body", "") or "") if (args.use_body and has_body_col) else ""
        # text + ticker search use the same content (title+body+lead when body present)
        content = (title + " " + body + " " + lead).strip() if body else (title + " " + lead).strip()
        if not content:
            continue
        # [MED-7 fix] ticker-match on FULL content (before truncation) so a ticker mentioned deep
        # in a body is not silently dropped; only the ENCODE input is char-capped for speed.
        ts = set(pat.findall(content))
        if not ts:
            continue
        # [perf] cap chars BEFORE tokenize: PhoBERT truncates to max_len tokens anyway, so capping
        # at ~max_len*6 chars is lossless for encoding and avoids tokenizing 500K-char bodies.
        content = content[: args.max_len * 6]
        if not ts:
            continue
        records.append({"date": d[:10], "tickers": ts, "text": content})
    print(f"[extract] input={Path(args.input).name} use_body={args.use_body} "
          f"ticker-matched: {len(records)}")
    if not records:
        print("[extract] nothing to do.")
        return

    # --- PhoBERT encode (frozen) ---
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"[extract] loading {args.model} (requires transformers<5)")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = [r["text"] for r in records]
    raw_dim = model.config.hidden_size  # 768 for phobert-base
    embs = np.zeros((len(records), raw_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True,
                      padding=True, max_length=args.max_len).to(device)
            cls = model(**enc).last_hidden_state[:, 0, :]   # [CLS] token
            embs[i:i + len(batch)] = cls.cpu().numpy()
            if (i // args.batch_size) % 20 == 0:
                print(f"  encoded {i+len(batch)}/{len(texts)}")
    # [MEDIUM-6 fix] fail loud if PhoBERT emitted NaN/Inf (corrupt Vietnamese text etc.)
    assert np.isfinite(embs).all(), \
        f"PhoBERT produced non-finite embeddings ({int(np.isnan(embs).sum())} NaN)"
    print(f"[extract] raw embeddings: {embs.shape}")

    # --- PCA (fit on train-period articles only) ---
    if args.use_pca and args.dim < raw_dim:
        from sklearn.decomposition import PCA
        train_mask = np.array([r["date"] < args.train_cutoff for r in records])
        n_train = int(train_mask.sum())
        # [HIGH-1 fix] NEVER widen PCA fit to val/test (silent leakage). Reduce dim instead.
        if n_train < args.dim:
            args.dim = max(1, n_train - 1)
            print(f"[warn] train articles ({n_train}) < dim; REDUCED dim to {args.dim} "
                  f"(train-only fit preserved, NO widening to val/test)")
        pca = PCA(n_components=args.dim).fit(embs[train_mask])
        embs = pca.transform(embs).astype(np.float32)
        print(f"[extract] PCA {raw_dim}->{args.dim} (fit on {int(train_mask.sum())} train articles, "
              f"explained var: {pca.explained_variance_ratio_.sum():.3f})")

    # --- Cache per-ticker: {date_str: [num_articles, dim]} ---
    by_ticker = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        for t in r["tickers"]:
            by_ticker[t][r["date"]].append(embs[i])

    n_written = 0
    for t in tickers:
        d = by_ticker.get(t)
        if not d:
            np.savez_compressed(out_dir / f"{t}_emb.npz")  # empty placeholder (no news)
            continue
        save = {date: np.stack(vecs, axis=0) for date, vecs in d.items()}
        np.savez_compressed(out_dir / f"{t}_emb.npz", **save)
        n_written += 1
    print(f"[extract] wrote {n_written}/{len(tickers)} ticker caches to {out_dir} "
          f"(dim={embs.shape[1]})")


if __name__ == "__main__":
    main()
