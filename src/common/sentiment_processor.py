"""
Sentiment Processor for Global Benchmark Datasets

Processes news articles to sentiment scores using FinBERT (English)
or PhoBERT (Vietnamese) depending on market.

Author: Stock Volatility Prediction Team
Date: 2026-08-01
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Bootstrap path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


SENTIMENT_MODELS = {
    "sp500": {
        "model": "ProsusAI/finbert",
        "language": "english",
    },
    "vn30": {
        "model": "vinai/phobert-base",
        "language": "vietnamese",
    },
}


def load_sentiment_model(market: str = "sp500"):
    """
    Load pre-trained sentiment model for the specified market.

    Args:
        market: Market identifier

    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    config = SENTIMENT_MODELS.get(market, {})
    if not config:
        raise ValueError(f"Unknown market: '{market}'. Available: {list(SENTIMENT_MODELS.keys())}")

    model_name = config["model"]
    print(f"[INFO] Loading sentiment model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return model, tokenizer


def score_sentiment(
    texts: list,
    model,
    tokenizer,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Score a list of texts for sentiment.

    Args:
        texts: List of text strings
        model: Pre-trained sentiment model
        tokenizer: Pre-trained tokenizer
        batch_size: Batch size for inference

    Returns:
        DataFrame with sentiment_score and sentiment_confidence
    """
    import torch
    from torch.nn.functional import softmax

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1)

        # FinBERT: labels are negative, neutral, positive
        # Score: -1 (negative) to +1 (positive)
        for j in range(len(batch)):
            neg = probs[j][0].item()
            neu = probs[j][1].item()
            pos = probs[j][2].item()

            # Composite score: pos - neg (range -1 to +1)
            score = pos - neg
            confidence = max(pos, neg, neu)

            results.append({
                "sentiment_score": score,
                "sentiment_confidence": confidence,
            })

    return pd.DataFrame(results)


def process_news_sentiment(
    market: str = "sp500",
    news_df: Optional[pd.DataFrame] = None,
    tickers: Optional[list] = None,
    output_dir: str = None,
):
    """
    Process news articles to sentiment scores per ticker.

    Args:
        market: Market identifier
        news_df: News DataFrame with columns: symbol, Title/Text, Publishdate
        tickers: List of tickers to process (default: all in news_df)
        output_dir: Directory to save sentiment CSVs
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, "data", "sentiment", market)

    os.makedirs(output_dir, exist_ok=True)

    # Load news if not provided
    if news_df is None:
        from datasets import load_dataset
        print("[INFO] Loading news dataset from Hugging Face")
        ds = load_dataset("KrossKinetic/SP500-Financial-News-Articles-Time-Series", split="train")
        news_df = ds.to_pandas()

    # Filter to requested tickers
    if tickers:
        news_df = news_df[news_df["symbol"].isin(tickers)]

    # Load model
    model, tokenizer = load_sentiment_model(market)

    # Process per ticker
    saved = 0
    for ticker in news_df["symbol"].unique():
        ticker_news = news_df[news_df["symbol"] == ticker]

        # Use Title + first 200 chars of Text as input
        texts = []
        for _, row in ticker_news.iterrows():
            title = row.get("Title", "")
            body = row.get("Text", "")[:200]
            texts.append(f"{title}. {body}")

        # Score sentiment
        scores = score_sentiment(texts, model, tokenizer)

        # Add date (use .values to avoid index alignment issues)
        scores["date"] = pd.to_datetime(ticker_news["Publishdate"].values).strftime("%Y-%m-%d")

        # Aggregate by date (daily average sentiment)
        daily = scores.groupby("date").agg({
            "sentiment_score": "mean",
            "sentiment_confidence": "mean",
        }).reset_index()
        daily["news_count"] = scores.groupby("date").size().values

        # Save
        output_file = os.path.join(output_dir, f"{ticker}_sentiment.csv")
        daily.to_csv(output_file, index=False)
        saved += 1
        print(f"[OK] {ticker}: {len(daily)} days -> {output_file}")

    print(f"\n[SUCCESS] Saved sentiment for {saved} tickers to {output_dir}")
