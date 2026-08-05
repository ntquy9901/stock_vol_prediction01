"""Compute Sentiment Decay State (Method 1, XU_LY_NEWS_THUA.md) per stock.

s_t = mask_t · score_t + (1 - mask_t) · s_{t-1} · decay
  mask=1 (has news): s_t = sentiment score (reset to fresh news)
  mask=0 (no news):  s_t = previous state × decay (carry-forward, exponentially fading)

Reads:  data/sentiment_baseline/{TICKER}_sentiment.csv (lexicon sentiment_1d, news_count_1d)
Writes: data/sentiment_decay/{TICKER}_sentiment.csv — SAME schema
        (sentiment_1d = decayed state, news_count_1d = mask 0/1)
        so the existing src/sentiment_baseline/ pipeline consumes it unchanged
        via --sentiment_dir data/sentiment_decay.

ISOLATED: read-only on sentiment_baseline data; writes to a separate decay dir.

Run:
  python baselines/2026-07-11_sentiment_decay/code/compute_decay.py
  python baselines/2026-07-11_sentiment_decay/code/compute_decay.py --decay 0.8
"""
import sys
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

IN_DIR = _ROOT / "data" / "sentiment_baseline"
OUT_DIR = _ROOT / "data" / "sentiment_decay"


def compute_decay_state(scores, masks, decay=0.9):
    """s_t = mask·score + (1-mask)·s_{t-1}·decay. Iterates chronologically.

    Args:
      scores: sentiment score per day (0 when no news — masked out).
      masks:  1 if news that day, else 0.
      decay:  carry-forward factor (0.9 = lose 10% per day).
    Returns: list of decayed states, same length as input.
    """
    s = 0.0
    states = []
    for score, mask in zip(scores, masks):
        s = float(score) if mask else s * decay
        states.append(s)
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decay", type=float, default=0.9, help="carry-forward decay factor")
    ap.add_argument("--in_dir", default=str(IN_DIR))
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_sentiment.csv"))
    print(f"[decay] {len(files)} sentiment files | decay={args.decay}")
    n_written = 0
    total_state_mass = 0.0  # diagnostic: how much non-zero state exists
    for f in files:
        df = pd.read_csv(f, dtype=str, keep_default_na=False)
        if "date" not in df.columns or "sentiment_1d" not in df.columns:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        scores = df["sentiment_1d"].astype(float).values
        # [MED-3 fix] news_count_1d always present in sentiment_baseline output — assert + use it
        # (avoids the `scores != 0` fallback which misclassifies net-neutral news days as no-news)
        assert "news_count_1d" in df.columns, f"{f.name}: missing news_count_1d column"
        masks = (df["news_count_1d"].astype(float) > 0).astype(int).values
        states = compute_decay_state(scores, masks, args.decay)
        total_state_mass += sum(abs(s) for s in states)
        out = pd.DataFrame({
            "date": df["date"],
            "sentiment_1d": states,                  # decayed state (only this column changes)
            "news_count_1d": df["news_count_1d"],    # [HIGH-1 fix] preserve original count, NOT mask
        })
        out.to_csv(out_dir / f.name, index=False)
        n_written += 1
    print(f"[decay] wrote {n_written} decay files to {out_dir} "
          f"(total |state| mass={total_state_mass:.1f})")


if __name__ == "__main__":
    main()
