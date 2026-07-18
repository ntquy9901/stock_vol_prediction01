"""Offline: extract PhoBERT embeddings for OBJECTIVE (khách quan) news -> cache .npz.

"Objective" = official corporate-event disclosures (vietstock/VSD, company_code already
tagged) + general news ticker-matched by regex (same pattern as the 2026-07-07 embedding
baseline's extract_embeddings.py) — as opposed to broker analyst reports ("báo cáo tổng hợp")
used by the existing news branch. See ../requirements/requirements.md and ../design/design.md.

ISOLATED: reads ONLY D:/bmad-projects/crawl_data/data/objective/*.csv (READ ONLY).
Writes ONLY data/objective_embedding/. No edits to src/ or sibling baselines.

ENV GOTCHA: requires `transformers<5` (4.57.x). See memory: project-sentiment-baseline-status.

Run:
  python baselines/2026-07-15_objective_news_baseline/code/extract_objective_embeddings.py
"""
import sys
import csv
import json
import pickle
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

OBJ_DIR = _ROOT.parent / "crawl_data" / "data" / "objective"
STOCK_DIR = _ROOT / "data" / "processed"
SUMMARY = STOCK_DIR / "processing_summary.csv"
OUT_DIR = _ROOT / "data" / "objective_embedding"
MANIFEST_NAME = "_manifest.json"   # {"document_ids": [...]} — rows already encoded, skipped on rerun
PCA_NAME = "_pca.pkl"              # fitted PCA persisted so PCA is NEVER refit on new (test-period)
                                    # data — refitting later would leak val/test into the transform.

TRAIN_CUTOFF = "2020-01-01"  # PCA fit only on records before this (train split)

# company_code already populated by the crawl -> use directly, no regex needed
DIRECT_FILES = ["vietstock_records.csv", "vsdc_records.csv"]
# company_code empty ("unenriched") -> ticker-regex match on title+raw_text
UNENRICHED_FILES = [
    "news_unenriched_vnexpress_records.csv",
    "news_unenriched_tuoitre_records.csv",
    "news_unenriched_thanhnien_records.csv",
    "news_unenriched_vietnamplus_records.csv",
    "news_unenriched_nld_records.csv",
]
# objective_v2026-07-*.csv intentionally SKIPPED: snapshot merge OF vietstock+vsdc
# themselves (double-count risk) — see requirements.md §2.

# General news mentions the BRAND/company name ("Vinamilk", "Vinfast"), almost never the
# ticker code ("VNM") -> ticker-regex alone matched only 2/402 unenriched rows. Manually
# curated brand-name aliases for VN30 (public, well-known Vietnamese companies) so general
# news can be routed to a stock WITHOUT the article containing the ticker code itself.
# Only aliases that DIFFER from the ticker string are listed (ticker-regex already covers
# the ticker-code case). Best-effort / not exhaustive — see requirements.md addendum.
NAME_ALIASES = {
    "ACB": ["Ngân hàng Á Châu"],
    "BCM": ["Becamex"],
    "BID": ["BIDV"],
    "BVH": ["Bảo Việt"],
    "CTG": ["VietinBank", "Vietinbank"],
    "GAS": ["PV Gas", "PetroVietnam Gas"],
    "GVR": ["Tập đoàn Cao su Việt Nam"],
    "HDB": ["HDBank"],
    "HPG": ["Hòa Phát"],
    "MBB": ["MBBank", "Ngân hàng Quân đội"],
    "MSN": ["Masan"],
    "MWG": ["Thế Giới Di Động", "Điện Máy Xanh", "Bách Hóa Xanh"],
    "NVL": ["Novaland"],
    "PDR": ["Phát Đạt"],
    "PLX": ["Petrolimex"],
    "POW": ["PV Power"],
    "SAB": ["Sabeco", "Bia Sài Gòn"],
    "SSB": ["SeABank"],
    "STB": ["Sacombank"],
    "TCB": ["Techcombank"],
    "TPB": ["TPBank"],
    "VCB": ["Vietcombank"],
    "VHM": ["Vinhomes"],
    "VIC": ["Vingroup", "Vinfast"],  # Vinfast is a Vingroup subsidiary -> routed to parent VIC
    "VJC": ["Vietjet"],
    "VNM": ["Vinamilk"],
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MAX_CHARS = 384  # cheap pre-tokenize cap (PhoBERT truncates to max_len tokens anyway)


def load_tickers():
    with open(SUMMARY, encoding="utf-8-sig") as f:
        return [r["ticker"].strip() for r in csv.DictReader(f) if (r.get("ticker") or "").strip()]


def _strip_html(s: str) -> str:
    return _HTML_TAG_RE.sub(" ", s or "")


def _clean_text(title: str, raw_text: str) -> str:
    title = (title or "").strip()
    raw_text = _strip_html(raw_text or "").strip()
    if raw_text and raw_text != title:
        return f"{title} {raw_text}".strip()
    return title


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(OBJ_DIR / name, dtype=str, encoding="utf-8-sig",
                       keep_default_na=False, on_bad_lines="skip")


def build_records(tickers, skip_ids=None):
    """Return list of {date, tickers:set, text, doc_id} for VN30-relevant objective rows.

    skip_ids: document_ids (or checksums) already encoded in a previous run — skipped so
    reruns only do work for NEW rows (incremental; see main()'s manifest handling). Rows
    with no document_id/checksum at all can't be tracked and are re-scanned every run
    (cheap: they're just skipped again by the same downstream filters).

    date = publish_time[:10]; rows with missing/malformed publish_time are dropped and
    counted separately (NOT silently folded into "0 ticker matches" in the log — e.g. all
    59/59 rows of news_unenriched_tuoitre_records.csv and 174/669 of vietstock_records.csv
    have no publish_time at all). [rejected alternative] falling back to crawl_time seemed
    safe (crawl_time always >= true disclosure time, no leakage) but was tried and REVERTED:
    it collapsed 179/298 test-period records onto a single calendar date (the crawl date),
    an artificial news-volume spike far worse than just losing that data — see
    requirements.md §5b addendum.

    Rows where publish_time > crawl_time are dropped ([leakage guard] a handful of
    vietstock rows carry a future record date, not the disclosure date — see
    requirements.md §3, ~0.7% of vietstock_records.csv).

    Cross-file dedup by document_id (falls back to checksum) — vietstock/vsdc rows all
    carry a document_id from the crawler, collisions are not expected but guarded anyway.

    Unenriched (general-news) rows are matched by ticker-code regex (case-SENSITIVE — VN30
    tickers are always written uppercase in Vietnamese financial press; case-insensitive
    would false-match common words e.g. "GAS" vs "gas") OR brand/company-name alias
    (NAME_ALIASES, case-insensitive — general news says "Vinamilk", not "VNM", so
    ticker-code alone matched almost nothing: 2/402, see requirements.md addendum 2026-07-15).
    """
    ticker_set = set(tickers)
    pat = re.compile(r"\b(" + "|".join(re.escape(t) for t in tickers) + r")\b")
    # ticker -> compiled alias pattern (brand/company name match), only for tickers we track
    alias_pats = {t: re.compile("|".join(re.escape(a) for a in aliases), re.IGNORECASE)
                  for t, aliases in NAME_ALIASES.items() if t in ticker_set}
    records = []
    seen_ids = set(skip_ids or ())

    def _dup_or_id(r):
        """Return (already_seen: bool, key: str) — key is '' when untrackable."""
        key = (getattr(r, "document_id", "") or getattr(r, "checksum", "") or "").strip()
        if not key:
            return False, ""
        if key in seen_ids:
            return True, key
        seen_ids.add(key)
        return False, key

    for name in DIRECT_FILES:
        path = OBJ_DIR / name
        if not path.exists():
            print(f"[skip] {name} not found")
            continue
        df = _read(name)
        n_kept = 0
        n_no_date = 0
        for r in df.itertuples(index=False):
            already, doc_id = _dup_or_id(r)
            if already:
                continue
            code = (getattr(r, "company_code", "") or "").strip().upper()
            if code not in ticker_set:
                continue
            pub = (getattr(r, "publish_time", "") or "")[:10]
            crawl = (getattr(r, "crawl_time", "") or "")[:10]
            if len(pub) < 10:
                n_no_date += 1
                continue
            if crawl and pub > crawl:
                continue
            text = _clean_text(getattr(r, "title", ""), getattr(r, "raw_text", ""))
            if not text:
                continue
            records.append({"date": pub, "tickers": {code}, "text": text[:_MAX_CHARS], "doc_id": doc_id})
            n_kept += 1
        print(f"[load] {name:42s} rows={len(df):>5d}  kept={n_kept}  no_date_dropped={n_no_date}")

    for name in UNENRICHED_FILES:
        path = OBJ_DIR / name
        if not path.exists():
            print(f"[skip] {name} not found")
            continue
        df = _read(name)
        n_kept = 0
        n_no_date = 0
        for r in df.itertuples(index=False):
            already, doc_id = _dup_or_id(r)
            if already:
                continue
            pub = (getattr(r, "publish_time", "") or "")[:10]
            crawl = (getattr(r, "crawl_time", "") or "")[:10]
            if len(pub) < 10:
                n_no_date += 1
                continue
            if crawl and pub > crawl:
                continue
            text = _clean_text(getattr(r, "title", ""), getattr(r, "raw_text", ""))
            if not text:
                continue
            ts = set(pat.findall(text))
            ts |= {t for t, ap_ in alias_pats.items() if ap_.search(text)}
            if not ts:
                continue
            records.append({"date": pub, "tickers": ts, "text": text[:_MAX_CHARS], "doc_id": doc_id})
            n_kept += 1
        print(f"[load] {name:42s} rows={len(df):>5d}  ticker/name-matched={n_kept}  "
              f"no_date_dropped={n_no_date}")

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vinai/phobert-base")
    ap.add_argument("--dim", type=int, default=64, help="PCA output dim (ignored if --no_pca)")
    ap.add_argument("--use_pca", dest="use_pca", action="store_true", default=True)
    ap.add_argument("--no_pca", dest="use_pca", action="store_false")
    ap.add_argument("--train_cutoff", default=TRAIN_CUTOFF)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--emb_dir", default=str(OUT_DIR))
    ap.add_argument("--refit_pca", action="store_true",
                    help="force refit PCA on this run's train-period records instead of reusing "
                         "the persisted PCA (only needed if NEW pre-train_cutoff data appears — "
                         "reusing new test-period data to refit would leak).")
    args = ap.parse_args()

    out_dir = Path(args.emb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST_NAME
    pca_path = out_dir / PCA_NAME
    tickers = load_tickers()
    print(f"[extract] VN30 tickers: {len(tickers)}")

    processed_ids = set()
    if manifest_path.exists():
        processed_ids = set(json.loads(manifest_path.read_text(encoding="utf-8"))["document_ids"])

    records = build_records(tickers, skip_ids=processed_ids)
    print(f"[extract] new records since last run: {len(records)} "
          f"(already processed: {len(processed_ids)})")
    if not records:
        print("[extract] nothing new to encode — caches already up to date.")
        return

    # --- PhoBERT encode (frozen) — only the NEW records ---
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"[extract] loading {args.model} (requires transformers<5)")
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
            if (i // args.batch_size) % 20 == 0:
                print(f"  encoded {i+len(batch)}/{len(texts)}")
    assert np.isfinite(embs).all(), \
        f"PhoBERT produced non-finite embeddings ({int(np.isnan(embs).sum())} NaN)"
    print(f"[extract] raw embeddings: {embs.shape}")

    # --- PCA: reuse the PERSISTED transform (never refit on new/test-period data — that
    # would leak). Only fit fresh on run #1 (no persisted PCA yet) or if --refit_pca forces it
    # (only safe to use if the NEW rows include pre-train_cutoff data). ---
    if args.use_pca and args.dim < raw_dim:
        from sklearn.decomposition import PCA
        if pca_path.exists() and not args.refit_pca:
            with open(pca_path, "rb") as f:
                pca = pickle.load(f)
            embs = pca.transform(embs).astype(np.float32)
            print(f"[extract] applied PERSISTED PCA ({pca_path.name}) {raw_dim}->{embs.shape[1]} "
                  f"(not refit — avoids leaking new/test-period data into the transform)")
        else:
            train_mask = np.array([r["date"] < args.train_cutoff for r in records])
            n_train = int(train_mask.sum())
            if n_train < 2:
                print(f"[warn] too few train-period records ({n_train}) for PCA; "
                      f"keeping raw {raw_dim}-d (no PCA persisted)")
            else:
                dim = args.dim
                if n_train < dim:
                    dim = max(1, n_train - 1)
                    print(f"[warn] train records ({n_train}) < dim; REDUCED dim to {dim} "
                          f"(train-only fit preserved, NO widening to val/test)")
                pca = PCA(n_components=dim).fit(embs[train_mask])
                embs = pca.transform(embs).astype(np.float32)
                with open(pca_path, "wb") as f:
                    pickle.dump(pca, f)
                print(f"[extract] PCA {raw_dim}->{dim} (fit on {n_train} train records, "
                      f"explained var: {pca.explained_variance_ratio_.sum():.3f}) -> "
                      f"persisted to {pca_path.name}")

    # --- Merge new records into existing per-ticker caches: {date_str: [num_articles, dim]} ---
    by_ticker = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(records):
        for t in r["tickers"]:
            by_ticker[t][r["date"]].append(embs[i])

    n_written = 0
    for t in tickers:
        new_d = by_ticker.get(t)
        cache_path = out_dir / f"{t}_emb.npz"
        existing = dict(np.load(cache_path, allow_pickle=False)) if cache_path.exists() else {}
        if not new_d:
            if not existing:
                np.savez_compressed(cache_path)  # empty placeholder (no news at all yet)
            continue  # nothing new for this ticker — existing cache (if any) is already correct
        merged = dict(existing)
        for date, vecs in new_d.items():
            arr = np.stack(vecs, axis=0)
            merged[date] = np.concatenate([merged[date], arr], axis=0) if date in merged else arr
        np.savez_compressed(cache_path, **merged)
        n_written += 1
    print(f"[extract] merged {n_written}/{len(tickers)} ticker caches in {out_dir} "
          f"(dim={embs.shape[1]})")

    # --- Update manifest: only rows with a trackable doc_id count toward "already processed" ---
    processed_ids |= {r["doc_id"] for r in records if r["doc_id"]}
    manifest_path.write_text(json.dumps({"document_ids": sorted(processed_ids)}, ensure_ascii=False),
                             encoding="utf-8")
    print(f"[extract] manifest updated: {len(processed_ids)} document_ids tracked")


if __name__ == "__main__":
    main()
