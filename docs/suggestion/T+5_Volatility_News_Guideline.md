# Guideline: Using News for T+5 Volatility Forecasting

## Objective

Predict **volatility at T+5** (or realized volatility over the next 5
trading days) without introducing data leakage.

------------------------------------------------------------------------

# 1. Correct Timeline

At prediction time:

``` text
T-30 ... T-2 T-1 T
                 ^
          Prediction time
```

Target:

``` text
T+1
T+2
T+3
T+4
T+5
```

Recommended label:

-   Realized Volatility (RV) over T+1\~T+5
-   or annualized volatility over the next 5 trading days

------------------------------------------------------------------------

# 2. News That May Be Used

Only use information available **at or before prediction time**.

Allowed:

``` text
News(T-7)
News(T-6)
...
News(T)
```

Never use:

``` text
News(T+1)
News(T+2)
...
```

Doing so causes **data leakage**.

------------------------------------------------------------------------

# 3. News Window

### Option 1 (Recommended)

Use the most recent **7 calendar days** of news.

``` text
Price: last 30 trading days
News: last 7 calendar days
```

This is the most common practical configuration.

### Option 2

Use 14 days of news.

Suitable for:

-   Macro events
-   Banking sector
-   Real estate
-   Long-lasting policy impacts

### Option 3

Use 30 days of news.

Often seen in Transformer-based research where the model learns
long-term dependencies.

------------------------------------------------------------------------

# 4. Don't Feed Every News Article

A single day may contain hundreds of articles.

Instead:

-   Filter by relevance
-   Keep only Top-K articles

Typical values:

-   Top 20
-   Top 30
-   Top 50

------------------------------------------------------------------------

# 5. How to Select Top-K

## Company relevance

Match using:

-   Stock ticker
-   Company name
-   CEO
-   Product/Brand

Example:

-   VCB
-   Vietcombank
-   Ngân hàng Vietcombank

------------------------------------------------------------------------

## Source quality

International:

-   Reuters
-   Bloomberg
-   Wall Street Journal
-   Nikkei

Vietnam:

-   CafeF
-   Vietstock
-   VnEconomy
-   NDH
-   SSI Research

------------------------------------------------------------------------

## Recency

Newer news should receive larger weights.

Example:

-   Today → 1.0
-   Yesterday → 0.9
-   Two days ago → 0.8

------------------------------------------------------------------------

# 6. Multiple News Per Day

## Mean Pooling

``` text
Article embeddings
        ↓
Average
        ↓
Daily embedding
```

Simple and effective.

## Attention Pooling

``` text
Multiple articles
        ↓
Transformer
        ↓
Attention
        ↓
Daily embedding
```

More powerful.

## Top-K Attention

``` text
Top-K news
      ↓
Attention
      ↓
Daily embedding
```

Common in recent research.

------------------------------------------------------------------------

# 7. Time Decay

Older news should gradually lose influence.

Example:

``` text
weight = exp(-λ × Δt)
```

Typical behavior:

Today:

-   weight = 1.0

Yesterday:

-   weight ≈ 0.9

Older:

-   progressively smaller.

------------------------------------------------------------------------

# 8. Different News Has Different Lifetimes

Examples:

CEO resignation

-   1--2 weeks

Quarterly earnings

-   1--3 days

Interest rate decisions

-   several weeks or months

Instead of manually defining these durations, modern Transformers often
learn them automatically.

------------------------------------------------------------------------

# 9. Recommended Input Configuration

## Price

30 trading days

Features:

-   OHLCV
-   HAR features
-   ATR
-   Historical volatility
-   Realized volatility

## News

7--14 calendar days

Only articles published before prediction time.

## Macro

Optional:

-   Interest rate
-   USD/VND
-   CPI
-   Commodity prices
-   Market indices

------------------------------------------------------------------------

# 10. Recommended Architecture

``` text
Historical Prices (30 trading days)
                │
      Time-Series Encoder
(PatchTST / Chronos / TimesFM)
                │
         Price Embedding
                │

News (last 7–14 days)
        │
Top-K Daily Articles
        │
Financial Language Model
(PhoBERT / FinBERT / ModernBERT)
        │
Attention Pooling
        │
Temporal Transformer
        │
News Embedding
        │
        └───────────────┐
                        ▼
             Cross Attention Fusion
       (Price ↔ News Interaction)
                        │
                        ▼
              MLP / Mixture of Experts
                        │
                        ▼
      Predict RV(T+1 ... T+5)
```

------------------------------------------------------------------------

# 11. Recommended Vietnamese Implementation

Given approximately five years of Vietnamese market data:

## Price Encoder

-   PatchTST
-   Chronos
-   TimesFM

## News Encoder

-   PhoBERT (recommended)
-   FinBERT (English)
-   ModernBERT
-   Financial LLM embeddings (optional)

## Fusion

Cross-Attention between price and news embeddings.

## Output

Predict:

-   Rolling 5-day Realized Volatility

instead of only predicting price direction.

------------------------------------------------------------------------

# 12. Best Practices

-   Never use future news.
-   Align labels carefully to avoid leakage.
-   Filter news using Top-K relevance.
-   Aggregate multiple articles into daily embeddings.
-   Apply time decay or let Transformers learn temporal importance.
-   Use Cross-Attention for price-news fusion.
-   Predict volatility (RV) rather than only price movement.
-   With only \~5 years of data, prefer compact architectures over very
    large LLMs.
