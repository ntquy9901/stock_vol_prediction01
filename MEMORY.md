# Memory Index - Stock Volatility Prediction VN30

**Last Updated:** 2026-08-08
**Purpose:** Index of project memory files for future reference

---

## Memory Files

### Project Memory
- **[Pooled LSTM, News, and GNN Pilot](memory/project_pooled_news_gnn_pilot_2026-08-08.md)** - Current architecture decisions, leakage invariants, worktree/commit state, and implementation continuation point
  - Date: 2026-08-08
  - Key: pooled asynchronous price samples for LSTM/news; synchronized dates only for graph tensors; GNN retained for ablation

- **[Sentiment Research Status](memory/project_sentiment_research_status.md)** - Current status of sentiment analysis integration research (deferred, baseline incomplete)
  - Date: 2026-06-28
  - Key: Date-based organization is standard, no Vietnamese dataset yet, baseline first

- **[Sentiment-Volatility Fusion SOTA](memory/project_sentiment_volatility_fusion_sota.md)** - SOTA architectures for combining sentiment with volatility (2025-2026)
  - Date: 2026-06-29
  - Key: MSGCA recommended, late fusion superior, per-stock per-day confirmed feasible

- **[Vietnam Datasets Status](memory/project_vietnam_datasets_status.md)** - Vietnamese stock news datasets availability and realistic approach
  - Date: 2026-06-29
  - Key: NO comprehensive dataset exists, build own dataset (4-5 weeks), vnstock to test first

### Feedback Memory
- **[Financial News Data Organization](memory/feedback_financial_news_data_organization.md)** - Mandatory patterns for financial news datasets
  - Date: 2026-06-28
  - Key: Per-ticker, per-date structure, time alignment, FNSPID pattern
  - Apply to: ALL financial news crawling projects

- **[Sentiment-Volatility Fusion Architectures](memory/feedback_sentiment_volatility_fusion_architectures.md)** - SOTA fusion patterns and implementation
  - Date: 2026-06-29
  - Key: Late fusion (0.876 vs 0.828), MSGCA architecture, complete pipeline
  - Apply to: ALL sentiment + time series fusion projects

- **[News Sparsity Solutions](memory/feedback_news_sparsity_solutions.md)** - Handling sparse news when creating daily features
  - Date: 2026-06-29
  - Key: Forward fill, exponential decay, days_since_last_news, 74% days have no news
  - Apply to: ALL per-day sentiment feature construction

### Reference Memory
- **[Sentiment Analysis Resources](memory/reference_sentiment_analysis_resources.md)** - Quick links to datasets, papers, tools
  - Date: 2026-06-28
  - Key: 29 sources categorized (FNSPID, Kaggle, arXiv, Vietnamese)
  - Use for: Dataset research, crawler implementation, sentiment models

---

## How to Use This Memory

**When Starting Sentiment Integration:**
1. Read `feedback_sentiment_volatility_fusion_architectures.md` for SOTA patterns
2. Check `feedback_financial_news_data_organization.md` for mandatory data patterns
3. Review `project_sentiment_volatility_fusion_sota.md` for architecture selection

**When Building Fusion Model:**
1. Read `feedback_sentiment_volatility_fusion_architectures.md` (MSGCA, LSTM-Transformer patterns)
2. Follow late fusion pattern (proven superior: 0.876 vs 0.828)
3. Implement per-stock, per-date architecture (confirmed feasible)

**When Building Crawler:**
1. Follow `feedback_financial_news_data_organization.md` architecture pattern
2. Reference `reference_sentiment_analysis_resources.md` (#9-15 for tutorials)
3. Validate against FNSPID implementation (#1-2 in reference)

**When Designing Dataset:**
1. Use date-based organization (feedback memory)
2. Check FNSPID structure (reference memory #1-2)
3. Validate alignment with HAR dataset (feedback memory)

---

## Key Takeaways

**✅ Confirmed Patterns (2026-06-28):**
- Date-based organization is industry standard (FNSPID gold standard)
- Per-ticker, per-date hierarchy is proven approach
- Time alignment with price data is mandatory

**✅ SOTA Architectures Identified (2026-06-29):**
- **MSGCA (2025):** Gated cross-attention, code available, TOP CHOICE
- **Late Fusion:** Proven superior (0.876 vs 0.828 accuracy, +5.8%)
- **LSTM-Transformer Hybrid:** Simple, matches current setup
- **Per-Stock Per-Day:** YES, confirmed feasible (FNSPID: 15.7M records)

**⚠️ Current Blockers:**
- Vietnamese historical news dataset not available
- Cafef.vn blocked (404, RSS 0% stock content)
- LSTM-GNN baseline incomplete (68.02% < 70% target)

**📋 Next Actions (When Ready):**
1. Complete LSTM-GNN baseline first (70%+ Dir Acc target)
2. Test vnstock package (2-3 days)
3. Implement MSGCA or LSTM-Transformer hybrid
4. Expected improvement: +2-5% Dir Acc, -5-10% RMSE

---

**For Full Research Context:**
- SOTA Fusion Architectures: `_bmad-output/planning-artifacts/research/technical-sentiment-volatility-fusion-sota-2026-06-29.md` (NEW!)
- Data Collection Research: `_bmad-output/planning-artifacts/research/technical-financial-news-crawling-dataset-research-2026-06-28.md`
- Vietnam Data Sources: `vietnam-stock-news-data-sources-plan.md`
- Previous Research: `_bmad-output/planning-artifacts/research/technical-lstm-gat-sentiment-analysis-vn30-research-2026-06-28.md`
