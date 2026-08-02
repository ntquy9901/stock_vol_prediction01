"""Process raw news -> per-stock DAILY sentiment files (ISOLATED baseline).

Simple design (per user): sentiment score + article count for THAT specific
day for THAT specific stock. No rolling, no coverage flag (the 22-day LSTM
input window already handles temporal context).

Reads:  data/raw/news/{news,data,data_archive}.csv
        data/processed/{TICKER}_processed.csv   (trading-day calendar, READ ONLY)
        data/processed/processing_summary.csv   (VN30 ticker list, READ ONLY)
Writes: data/sentiment_baseline/{TICKER}_sentiment.csv
        cols: date, sentiment_1d, news_count_1d

Does NOT modify data/processed/ or any existing data/folder.

Run:  python -m src.sentiment_baseline.process_news_to_sentiment
"""
import csv
import re
from pathlib import Path
from collections import defaultdict

from .lexicon import score as lexicon_score

# Project root (src/sentiment_baseline/ -> up 3)
ROOT = Path(__file__).resolve().parent.parent.parent
NEWS_DIR = ROOT / 'data' / 'raw' / 'news'
STOCK_DIR = ROOT / 'data' / 'processed'
OUT_DIR = ROOT / 'data' / 'sentiment_baseline'

NEWS_FILES = ['data.csv', 'data_2021_2025.csv', 'data_archive.csv']


def load_vn30_tickers():
    """Load 30 VN30 tickers from processing_summary.csv (read only)."""
    summary = STOCK_DIR / 'processing_summary.csv'
    tickers = []
    with open(summary, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            t = (row.get('ticker') or '').strip()
            if t:
                tickers.append(t)
    return tickers


def parse_date(s):
    """Parse DD/MM/YYYY -> 'YYYY-MM-DD'. Handle 2-digit years. None if invalid."""
    try:
        dd, mm, yyyy = s.strip().split('/')
        y = int(yyyy)
        y = 2000 + y if y < 100 else y
        if not (1990 <= y <= 2100):
            return None
        return f"{y:04d}-{int(mm):02d}-{int(dd):02d}"
    except Exception:
        return None


def find_tickers(title, ticker_res):
    return [t for t, rx in ticker_res.items() if rx.search(title)]


def load_dedup_news():
    """Load + dedup the 3 news files by id."""
    seen = {}
    for fname in NEWS_FILES:
        path = NEWS_DIR / fname
        if not path.exists():
            print(f"[news] (skip, not found) {path}")
            continue
        with open(path, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                rid = row.get('id', '')
                if rid and rid not in seen:
                    seen[rid] = row
    return list(seen.values())


def main(out_dir=None, scorer='lexicon', model=None):
    import pandas as pd

    out_dir_path = Path(out_dir) if out_dir else OUT_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)
    tickers = load_vn30_tickers()
    ticker_res = {t: re.compile(r'\b' + t + r'\b') for t in tickers}
    print(f"[news] VN30 tickers: {len(tickers)}")

    articles = load_dedup_news()
    print(f"[news] unique articles (dedup by id): {len(articles)}")

    # Score all titles up-front (batch for transformer scorers)
    titles = [a.get('title', '') or '' for a in articles]
    if scorer == 'phobert':
        from .phobert_scorer import score_batch
        print(f"[news] scorer=phobert, model={model or 'default'}; "
              f"scoring {len(titles)} titles (batch)...")
        scores = score_batch(titles, model_name=model)
    else:
        scores = [lexicon_score(t) for t in titles]
    print(f"[news] scorer={scorer}; scored {len(scores)} titles")

    # Dispatch to tickers per day (also keep titles for review)
    daily = defaultdict(lambda: defaultdict(list))        # ticker -> date -> [scores]
    daily_titles = defaultdict(lambda: defaultdict(list))  # ticker -> date -> [titles]
    n_matched = 0
    for i, a in enumerate(articles):
        title = a.get('title', '') or ''
        d = parse_date(a.get('date', ''))
        if not d:
            continue
        ts = find_tickers(title, ticker_res)
        if not ts:
            continue
        n_matched += 1
        s = scores[i]
        for t in ts:
            daily[t][d].append(s)
            daily_titles[t][d].append(title)

    print(f"[news] articles matching >=1 VN30 ticker: {n_matched} "
          f"({n_matched * 100 // max(1, len(articles))}%)")

    # Per-stock: align to trading days (left join), write daily sentiment + count
    n_written = 0
    for ticker in tickers:
        stock_file = STOCK_DIR / f"{ticker}_processed.csv"
        if not stock_file.exists():
            print(f"[warn] no stock file for {ticker}, skipping")
            continue
        df = pd.read_csv(stock_file, usecols=['date'])
        df['date'] = df['date'].astype(str)

        rows = []
        for d, sc in daily.get(ticker, {}).items():
            tit = daily_titles.get(ticker, {}).get(d, [])
            rows.append({
                'date': d,
                'sentiment_1d': sum(sc) / len(sc),
                'news_count_1d': len(sc),
                'news_titles': (' | '.join(tit))[:1000],  # cap for CSV readability
            })
        if rows:
            sent_df = pd.DataFrame(rows)
            df = df.merge(sent_df, on='date', how='left')
            df['sentiment_1d'] = df['sentiment_1d'].fillna(0.0)
            df['news_count_1d'] = df['news_count_1d'].fillna(0)
            df['news_titles'] = df['news_titles'].fillna('')
        else:
            df['sentiment_1d'] = 0.0
            df['news_count_1d'] = 0
            df['news_titles'] = ''

        out = df[['date', 'sentiment_1d', 'news_count_1d', 'news_titles']]
        out.to_csv(out_dir_path / f"{ticker}_sentiment.csv", index=False)
        n_written += 1

    print(f"[news] wrote {n_written} sentiment files to {out_dir_path}")
    print(f"[news] done.")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default=None, help='Output folder (default: data/sentiment_baseline)')
    p.add_argument('--scorer', default='lexicon', choices=['lexicon', 'phobert'],
                   help='Sentiment scorer (default: lexicon). phobert uses a HF transformer model')
    p.add_argument('--model', default=None,
                   help='HF model name for --scorer phobert (default: cardiffnlp/twitter-xlm-roberta-base-sentiment)')
    args = p.parse_args()
    main(out_dir=args.out_dir, scorer=args.scorer, model=args.model)
