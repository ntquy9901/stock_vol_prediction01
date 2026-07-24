# Feature Engineering từ tin tức cho phân tích và dự báo cổ phiếu

> Tài liệu tổng hợp các nhóm feature thường được sử dụng trong nghiên cứu quốc tế về financial news, stock movement và volatility forecasting.  
> Mục tiêu ứng dụng: xây dựng news features cho Ridge, GBM, HAR/HAR-X và GNN, đặc biệt cho các horizon T+1, T+5 và T+10.  
> Cập nhật link paper: 18/07/2026.

---

## 1. Kết luận chính từ literature

Các nghiên cứu từ AZFinText đến các mô hình event-based, knowledge graph, FinBERT và FININ cho thấy:

- Sentiment `positive/negative/neutral` đơn thuần thường chưa khai thác hết thông tin trong tin tức.
- Cần xác định đúng **công ty mục tiêu** và mức độ liên quan của tin đối với công ty đó.
- Cấu trúc sự kiện như actor, action, object, time và event type thường giàu thông tin hơn Bag-of-Words thuần.
- Tin mới, tin được nhiều nguồn đăng và tin có mức độ bất định cao thường liên quan mạnh hơn đến market reaction.
- Tác động tin tức có thể có độ trễ và kéo dài nhiều ngày, vì vậy không nên chỉ sử dụng tin của đúng ngày dự báo.
- Contextual embeddings chứa thông tin vượt ra ngoài sentiment, nhưng phải kiểm soát số chiều khi dataset nhỏ.
- Quan hệ giữa các công ty, ngành, chuỗi cung ứng và lãnh đạo có thể tạo spillover giữa các cổ phiếu.
- News features nên được kết hợp với giá, volume, realized volatility và trạng thái thị trường.

Bộ feature tổng quát nên kết hợp:

```text
Relevance
+ Target-specific sentiment
+ Uncertainty
+ Fine-grained event
+ Novelty
+ Dissemination
+ Temporal decay
+ Semantic embedding
+ Cross-stock spillover
+ Market state
```

---

# 2. Entity và stock relevance features

Trước khi đo sentiment hoặc event impact, cần xác định bài báo có thực sự liên quan đến cổ phiếu mục tiêu hay không.

## 2.1 Feature đề xuất

| Feature | Kiểu | Ý nghĩa |
|---|---:|---|
| `target_ticker` | categorical | Mã cổ phiếu mục tiêu |
| `target_company` | categorical | Tên doanh nghiệp chuẩn hóa |
| `company_relevance_score` | float | Mức liên quan của bài viết với doanh nghiệp |
| `company_mention_count` | integer | Số lần công ty được nhắc |
| `headline_contains_target` | binary | Công ty có xuất hiện trong tiêu đề |
| `first_mention_position` | float | Vị trí xuất hiện đầu tiên trong bài |
| `is_primary_company` | binary | Công ty có phải chủ thể chính |
| `actor_is_target` | binary | Công ty mục tiêu là bên thực hiện hành động |
| `object_is_target` | binary | Công ty mục tiêu là bên chịu tác động |
| `related_company_count` | integer | Số doanh nghiệp khác cùng xuất hiện |
| `industry` | categorical | Ngành của doanh nghiệp |
| `country_market` | categorical | Quốc gia hoặc thị trường liên quan |

## 2.2 Vì sao nhóm feature này quan trọng?

Một bài báo có thể nhắc nhiều doanh nghiệp nhưng không tác động giống nhau đến từng doanh nghiệp.

Ví dụ:

> Công ty A giành được hợp đồng lớn trước đối thủ B.

Diễn giải hợp lý:

```text
A:
    actor_or_winner = 1
    target_sentiment > 0

B:
    competitor_or_loser = 1
    target_sentiment < 0
```

Nếu chỉ sử dụng document-level sentiment, mô hình có thể gán cùng một sentiment cho cả A và B.

## 2.3 Cách tính relevance đơn giản

Có thể khởi đầu bằng một công thức heuristic:

```text
relevance_score =
      0.35 × headline_contains_target
    + 0.20 × lead_paragraph_contains_target
    + 0.20 × normalized_mention_count
    + 0.15 × actor_or_object_is_target
    + 0.10 × source_metadata_matches_target
```

Sau đó calibrate bằng một tập dữ liệu được gán nhãn:

```text
direct
indirect
irrelevant
```

## Paper liên quan

- Schumaker & Chen, **Textual Analysis of Stock Market Prediction Using Breaking Financial News: The AZFin Text System**  
  DOI: https://doi.org/10.1145/1462198.1462204

- Ding et al., **Using Structured Events to Predict Stock Price Movement: An Empirical Investigation**  
  Paper: https://aclanthology.org/D14-1148/  
  PDF: https://aclanthology.org/D14-1148.pdf

---

# 3. Sentiment features

## 3.1 Không chỉ lưu một sentiment label

Nên lưu đầy đủ xác suất:

```text
positive_probability
negative_probability
neutral_probability
sentiment_score
sentiment_confidence
sentiment_extremeness
```

Công thức thường dùng:

```text
sentiment_score = positive_probability - negative_probability

sentiment_extremeness = abs(sentiment_score)

sentiment_confidence =
    max(
        positive_probability,
        negative_probability,
        neutral_probability
    )
```

## 3.2 Financial dictionary features

Từ vựng tài chính khác với ngôn ngữ phổ thông. Một số từ có vẻ tiêu cực trong từ điển thông thường nhưng lại mang nghĩa trung tính trong báo cáo tài chính.

Có thể bổ sung các nhóm từ trong Loughran–McDonald:

```text
lm_positive_ratio
lm_negative_ratio
lm_uncertainty_ratio
lm_litigious_ratio
lm_constraining_ratio
lm_strong_modal_ratio
lm_weak_modal_ratio
```

Các feature ngôn ngữ bổ sung:

```text
negation_count
negated_positive_count
negated_negative_count
uncertainty_phrase_count
forward_looking_statement_count
```

## 3.3 Target-specific sentiment

Nên phân biệt:

```text
document_sentiment
target_company_sentiment
event_sentiment
sector_sentiment
market_sentiment
```

Một bài viết tích cực cho công ty A có thể tiêu cực cho đối thủ B. Vì vậy, sentiment cần gắn với:

```text
(company, event, sentiment)
```

hoặc chi tiết hơn:

```text
(company, industry, coarse_event, fine_event, sentiment)
```

## 3.4 Sentiment feature cho return và volatility

### Đối với return direction

Các feature có dấu thường hợp lý hơn:

```text
positive_score
negative_score
signed_sentiment
target_sentiment
```

### Đối với volatility

Các feature đo cường độ và bất định thường phù hợp hơn:

```text
abs_sentiment
sentiment_extremeness
negative_probability
uncertainty_score
sentiment_dispersion
source_disagreement
event_severity
```

Diễn giải:

```text
signed sentiment       → hướng return
absolute sentiment     → cường độ cú sốc
uncertainty            → mức rủi ro
sentiment disagreement → bất đồng thông tin
```

## Paper liên quan

- Loughran & McDonald, **When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks**  
  DOI: https://doi.org/10.1111/j.1540-6261.2010.01625.x  
  PDF của nhà xuất bản: https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1540-6261.2010.01625.x

- Araci, **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models**  
  Paper: https://arxiv.org/abs/1908.10063  
  PDF: https://arxiv.org/pdf/1908.10063

- Chen et al., **EFSA: Towards Event-Level Financial Sentiment Analysis**  
  Paper: https://arxiv.org/abs/2404.08681  
  PDF: https://arxiv.org/pdf/2404.08681

---

# 4. Event features

Event features là một trong các nhóm feature quan trọng nhất đối với stock movement và volatility.

## 4.1 Structured event representation

Ding et al. biểu diễn sự kiện theo dạng:

```text
Event = (Actor, Action, Object, Time)
```

Ví dụ:

```text
Actor  = Company A
Action = acquire
Object = Company B
Time   = publication/event timestamp
```

Mở rộng thực tế:

```text
Event = (
    actor,
    action,
    object,
    event_type,
    event_subtype,
    value,
    unit,
    status,
    certainty,
    time
)
```

## 4.2 Event types nên extract

### Earnings và tài chính

```text
earnings_result
earnings_surprise
revenue_growth
profit_warning
earnings_guidance
guidance_raise
guidance_cut
cash_flow_change
margin_change
```

### Corporate actions

```text
dividend
share_buyback
share_issuance
capital_increase
stock_split
insider_transaction
```

### M&A và hợp tác

```text
merger
acquisition
divestiture
joint_venture
strategic_partnership
contract_award
contract_loss
```

### Sản phẩm và vận hành

```text
new_product
product_delay
product_recall
factory_opening
factory_shutdown
capacity_expansion
supply_disruption
```

### Quản trị doanh nghiệp

```text
ceo_change
management_change
board_change
auditor_change
corporate_governance_issue
```

### Pháp lý và rủi ro

```text
lawsuit
regulatory_investigation
fraud_investigation
penalty
credit_rating_upgrade
credit_rating_downgrade
debt_default
bankruptcy
```

### Macro và ngành

```text
interest_rate_change
exchange_rate_change
commodity_price_change
tax_policy
industry_regulation
trade_policy
geopolitical_event
```

## 4.3 Event attributes

```text
event_type
event_subtype
event_trigger
event_actor
event_object
event_value
event_currency
event_direction
event_status
event_certainty
event_severity
event_scope
```

Ví dụ `event_status`:

```text
rumor
considering
planned
announced
approved
in_progress
completed
cancelled
denied
```

Ví dụ `event_certainty`:

```text
confirmed
probable
possible
speculative
```

## 4.4 Event severity

Có thể xây dựng severity score:

```text
event_severity =
    event_type_base_weight
    × relevance_score
    × certainty_weight
    × magnitude_weight
```

Ví dụ:

```text
magnitude_weight =
    abs(event_value) / company_scale
```

Trong đó `company_scale` có thể là:

```text
market_cap
annual_revenue
total_assets
average_daily_volume
```

## Paper liên quan

- Ding et al., **Using Structured Events to Predict Stock Price Movement: An Empirical Investigation**  
  https://aclanthology.org/D14-1148/

- Chen et al., **Incorporating Fine-grained Events in Stock Movement Prediction**  
  Paper: https://arxiv.org/abs/1910.05078  
  PDF: https://arxiv.org/pdf/1910.05078

- Chen et al., **EFSA: Towards Event-Level Financial Sentiment Analysis**  
  https://arxiv.org/abs/2404.08681

---

# 5. Semantic embedding features

## 5.1 Các embedding thường dùng

```text
headline_embedding
summary_embedding
body_embedding
event_embedding
target_context_embedding
```

Model phổ biến:

```text
BERT
FinBERT
Sentence-BERT
domain-specific financial language model
multilingual sentence transformer
```

Với tiếng Việt có thể thử:

```text
PhoBERT
multilingual-e5
multilingual sentence-transformer
Vietnamese financial model được fine-tune riêng
```

## 5.2 Không nên đưa 768 chiều trực tiếp vào model nhỏ

Với dữ liệu khoảng năm năm và news thưa, vector 768 chiều dễ gây:

- Overfitting.
- Multicollinearity.
- Tỷ lệ số feature trên số sample quá cao.
- Mô hình khó giải thích.
- Kết quả thiếu ổn định qua từng time split.

Các cách giảm chiều:

```text
PCA: 16, 32 hoặc 64 chiều
Autoencoder: 16–64 chiều
Supervised projection
Embedding clustering
Topic probability
Feature selection theo walk-forward validation
```

Ví dụ schema:

```text
embedding_pca_01
embedding_pca_02
...
embedding_pca_32
```

## 5.3 Các embedding similarity feature hữu ích

```text
similarity_to_previous_news
similarity_to_recent_company_news
similarity_to_sector_news
similarity_to_known_event_centroid
distance_to_positive_event_centroid
distance_to_negative_event_centroid
```

## Paper liên quan

- Chen, **Stock Movement Prediction with Financial News Using Contextualized Embedding from BERT**  
  Paper: https://arxiv.org/abs/2107.08721  
  PDF: https://arxiv.org/pdf/2107.08721

- Araci, **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models**  
  https://arxiv.org/abs/1908.10063

- Guo & Hauptmann, **Fine-Tuning Large Language Models for Stock Return Prediction Using Newsflow**  
  Paper: https://arxiv.org/abs/2407.18103  
  PDF: https://arxiv.org/pdf/2407.18103

---

# 6. Topic và news-category features

Không phải loại tin nào cũng tác động như nhau.

Ví dụ:

- Earnings surprise có thể ảnh hưởng trực tiếp đến valuation.
- Credit downgrade có thể làm tăng risk và volatility.
- Tin CSR hoặc hoạt động cộng đồng có thể ít tác động ngắn hạn.
- M&A, phá sản hoặc điều tra pháp lý thường có impact mạnh hơn.

## 6.1 Feature đề xuất

```text
topic_earnings_probability
topic_ma_probability
topic_regulation_probability
topic_product_probability
topic_management_probability
topic_credit_probability
topic_litigation_probability
topic_macro_probability
topic_governance_probability
topic_supply_chain_probability
```

Không nên chỉ lưu một `topic_id`. Nên lưu xác suất của các topic:

```text
top_topic_id
top_topic_probability
second_topic_id
second_topic_probability
topic_entropy
```

Công thức entropy:

```text
topic_entropy = -Σ p(topic_i) × log(p(topic_i))
```

Topic entropy cao có thể cho thấy bài viết chứa nhiều chủ đề hoặc khó phân loại.

## 6.2 Hai hướng xây dựng topic

### Supervised topic classification

Sử dụng taxonomy event/topic cố định:

```text
earnings
ma
credit
regulation
management
product
legal
macro
```

Ưu điểm:

- Dễ giải thích.
- Dễ aggregate.
- Dễ kiểm tra bằng ablation.

### Unsupervised hoặc weakly supervised

```text
embedding
→ clustering
→ cluster/topic label
→ kiểm tra market reaction của từng cluster
```

Ưu điểm:

- Tìm được loại tin chưa có trong taxonomy.
- Phù hợp exploratory analysis.

## Paper liên quan

- Feuerriegel & Pröllochs, **Investor Reaction to Financial Disclosures Across Topics: An Application of Latent Dirichlet Allocation**  
  Paper: https://arxiv.org/abs/1805.03308  
  PDF: https://arxiv.org/pdf/1805.03308

- Scherrmann, **Multi-Label Topic Model for Financial Textual Data**  
  Paper: https://arxiv.org/abs/2311.07598  
  PDF: https://arxiv.org/pdf/2311.07598

---

# 7. Novelty và duplicate features

Một bài báo đăng lại thông tin cũ thường có impact thấp hơn bài đầu tiên công bố sự kiện.

Điều này đặc biệt quan trọng với dữ liệu Việt Nam, vì cùng một thông cáo có thể được nhiều báo đăng lại gần như nguyên văn.

## 7.1 Feature đề xuất

```text
novelty_1d
novelty_3d
novelty_7d
novelty_30d

max_similarity_previous_1d
max_similarity_previous_7d

similar_article_count
duplicate_cluster_size
is_first_report
hours_since_first_report
fresh_news_ratio
```

## 7.2 Công thức novelty

```text
novelty_7d =
    1 - max_cosine_similarity(
        current_article_embedding,
        previous_company_articles_in_last_7_days
    )
```

Có thể tính riêng:

```text
headline_novelty
body_novelty
event_novelty
```

## 7.3 Duplicate clustering

Quy trình:

```text
normalize text
→ remove boilerplate
→ embedding
→ similarity matching
→ cluster bài cùng sự kiện
```

Mỗi cluster chỉ nên đóng góp một event chính, sau đó dùng số lượng bài và nguồn như dissemination features.

## Paper liên quan

- Mizuno et al., **The Impact of the Novelty and Topicality of Business News on Financial Markets**  
  Paper: https://arxiv.org/abs/1507.06477  
  PDF: https://arxiv.org/pdf/1507.06477

---

# 8. Dissemination, coverage và attention features

Một sự kiện được nhiều nguồn độc lập đăng trong thời gian ngắn có thể có phạm vi tiếp cận và market attention cao hơn.

## 8.1 Feature đề xuất

```text
unique_source_count
article_cluster_size
news_volume_1h
news_volume_6h
news_volume_1d
news_volume_3d
coverage_breadth
coverage_acceleration
source_growth_rate
```

Ví dụ:

```text
coverage_acceleration =
    article_count_last_6h
    - article_count_previous_6h
```

```text
coverage_breadth =
    unique_source_count / total_sources_monitored
```

## 8.2 Source quality

Có thể thêm:

```text
source_reliability_score
official_source_flag
exchange_disclosure_flag
company_press_release_flag
major_news_agency_flag
anonymous_source_flag
```

Không nên xem 20 bài copy từ cùng một nguồn là 20 nguồn độc lập.

Cần phân biệt:

```text
article_count
unique_source_count
unique_original_source_count
```

## Paper liên quan

- FinGPT-related work, **Enhancing Sentiment-Based Stock Movement Prediction by Incorporating News Dissemination**  
  Paper: https://arxiv.org/abs/2412.10823  
  PDF: https://arxiv.org/pdf/2412.10823

- Mizuno et al., **The Impact of the Novelty and Topicality of Business News on Financial Markets**  
  https://arxiv.org/abs/1507.06477

---

# 9. Temporal và decay features

## 9.1 Publication-time features

```text
publication_timestamp
publication_hour
publication_day_of_week
is_weekend
is_holiday
is_pre_market
is_during_market
is_after_market
minutes_to_market_open
minutes_to_market_close
```

## 9.2 Event-time và publication-time

Nên phân biệt:

```text
event_time
announcement_time
publication_time
first_crawl_time
```

Một bài báo có thể được xuất bản lại sau khi sự kiện đã xảy ra nhiều giờ hoặc nhiều ngày.

## 9.3 Trading-day alignment

Ví dụ với thị trường đóng cửa:

```text
Tin trước giờ đóng cửa:
    có thể gắn vào trading day hiện tại

Tin sau giờ đóng cửa:
    nên gắn vào trading day kế tiếp
```

Phải định nghĩa rõ cutoff và tránh look-ahead leakage.

## 9.4 Time decay

```text
time_decay = exp(-lambda × age_hours)
```

News score có trọng số:

```text
decayed_sentiment =
    relevance
    × novelty
    × sentiment
    × time_decay
```

```text
decayed_event_impact =
    relevance
    × novelty
    × event_severity
    × certainty
    × time_decay
```

Có thể thử nhiều half-life:

```text
6 hours
1 day
3 days
5 trading days
10 trading days
```

## 9.5 Delayed pricing và long memory

Không nên chỉ dùng tin trong ngày. Tạo nhiều cửa sổ:

```text
news_window_1d
news_window_3d
news_window_5d
news_window_10d
news_window_20d
```

Gợi ý theo horizon:

```text
T+1:
    1d, 2d, 3d

T+5:
    3d, 5d, 10d

T+10:
    5d, 10d, 20d
```

## Paper liên quan

- Wang, Cohen & Ma, **Modeling News Interactions and Influence for Financial Market Prediction (FININ)**  
  ACL Anthology: https://aclanthology.org/2024.findings-emnlp.189/  
  arXiv: https://arxiv.org/abs/2410.10614  
  PDF: https://arxiv.org/pdf/2410.10614

- Xu & Cohen, **Stock Movement Prediction from Tweets and Historical Prices (StockNet)**  
  Paper: https://aclanthology.org/P18-1183/  
  PDF: https://aclanthology.org/P18-1183.pdf

---

# 10. News interaction và aggregation features

Khi một cổ phiếu có nhiều tin trong ngày, không nên chỉ tính trung bình đơn giản.

## 10.1 Daily aggregation cơ bản

Theo `ticker × trading_day`:

```text
news_count
unique_source_count
positive_count
negative_count
neutral_count

sentiment_mean
sentiment_median
sentiment_max
sentiment_min
sentiment_std

sentiment_extremeness_mean
sentiment_extremeness_max

uncertainty_mean
uncertainty_max

relevance_mean
relevance_max

novelty_mean
novelty_max

event_severity_mean
event_severity_max
```

## 10.2 Dispersion và disagreement

```text
sentiment_std
sentiment_interquartile_range
source_sentiment_std
positive_negative_balance
sentiment_disagreement
topic_entropy
event_type_diversity
```

Ví dụ:

```text
positive_negative_balance =
    positive_count - negative_count
```

```text
sentiment_disagreement =
    1 - abs(mean_signed_sentiment)
```

Nên kết hợp disagreement với sentiment extremeness để tránh trường hợp tất cả bài đều neutral.

## 10.3 Weighted aggregation

```text
article_weight =
    relevance
    × novelty
    × source_reliability
    × time_decay
```

```text
weighted_sentiment =
    Σ(article_weight_i × sentiment_i)
    / Σ(article_weight_i)
```

Tương tự cho:

```text
weighted_uncertainty
weighted_event_severity
weighted_embedding
```

## 10.4 Attention-based aggregation

Với neural model, có thể encode từng bài riêng rồi học attention weight:

```text
article embeddings
→ attention
→ daily news representation
```

Cách này tránh việc một tin quan trọng bị chìm trong nhiều tin ít liên quan.

## Paper liên quan

- Wang, Cohen & Ma, **FININ**  
  https://aclanthology.org/2024.findings-emnlp.189/

- Chen, **FT-CE-RNN**  
  https://arxiv.org/abs/2107.08721

---

# 11. Relational và spillover features

Tin của một doanh nghiệp có thể ảnh hưởng đến các doanh nghiệp khác.

## 11.1 Quan hệ có thể sử dụng

```text
same_industry
competitor
supplier
customer
parent_company
subsidiary
joint_venture
major_shareholder
shared_executive
shared_director
shared_institutional_owner
co_mentioned
correlated_price
```

## 11.2 Feature đề xuất

```text
sector_news_sentiment
competitor_news_sentiment
supplier_news_sentiment
customer_news_sentiment

related_company_event_severity
related_company_news_count
co_mention_count
relationship_strength

neighbor_sentiment_mean
neighbor_uncertainty_mean
neighbor_event_severity_max
```

## 11.3 Dùng trong GNN

Node:

```text
stock
company
executive
industry
```

Edge:

```text
belongs_to_industry
competitor_of
supplier_of
customer_of
managed_by
co_mentioned_with
```

Node feature theo ngày:

```text
OHLCV/HAR features
news sentiment
event features
novelty
news volume
embedding
```

GNN có thể học news spillover giữa các công ty thay vì đưa tất cả feature vào một vector phẳng.

## Paper liên quan

- Zhao et al., **Stock Movement Prediction Based on Bi-typed Hybrid-relational Market Knowledge Graph via Dual Attention Networks**  
  Paper: https://arxiv.org/abs/2201.04965  
  PDF: https://arxiv.org/pdf/2201.04965

- Kim et al., **HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction**  
  Paper: https://arxiv.org/abs/1908.07999  
  PDF: https://arxiv.org/pdf/1908.07999

- Sawhney et al., **Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations**  
  Paper: https://aclanthology.org/2020.emnlp-main.676/  
  PDF: https://aclanthology.org/2020.emnlp-main.676.pdf

---

# 12. Market-state và price interaction features

News impact phụ thuộc trạng thái hiện tại của cổ phiếu và thị trường.

## 12.1 Price và liquidity controls

```text
lagged_return_1d
lagged_return_5d
lagged_return_20d

realized_vol_1d
realized_vol_5d
realized_vol_22d

volume_zscore
turnover
bid_ask_spread
market_cap
liquidity_bucket
```

## 12.2 Market regime

```text
market_return
market_volatility
sector_return
sector_volatility
bull_bear_regime
high_low_vol_regime
```

## 12.3 Interaction features

```text
sentiment × relevance
sentiment × novelty
sentiment × market_regime
uncertainty × current_volatility
event_severity × liquidity
negative_news × leverage
sector_news × stock_sector_exposure
```

Các interaction này đặc biệt phù hợp cho GBM, XGBoost hoặc LightGBM.

---

# 13. Feature set đề xuất cho bài toán volatility

## 13.1 Minimum Viable Feature Set

Bộ đầu tiên nên đủ nhỏ để kiểm tra signal:

```text
news_count
unique_source_count

relevance_mean
relevance_max

positive_probability_mean
negative_probability_mean
sentiment_score_mean
sentiment_extremeness_mean
sentiment_extremeness_max
sentiment_std

uncertainty_mean
uncertainty_max

novelty_mean
novelty_max
duplicate_cluster_size_max

event_severity_mean
event_severity_max
event_type_count

hours_since_latest_news
recency_weighted_sentiment
recency_weighted_uncertainty
recency_weighted_event_severity
```

## 13.2 Advanced Feature Set

```text
embedding_pca_01 ... embedding_pca_32

sector_sentiment
competitor_sentiment
related_company_sentiment

coverage_breadth
coverage_acceleration
fresh_news_ratio
first_report_ratio

source_sentiment_std
event_type_diversity
topic_entropy
```

## 13.3 Feature ưu tiên theo target

### Dự báo return direction

Ưu tiên:

```text
signed_sentiment
positive_probability
negative_probability
target_sentiment
event_direction
earnings_surprise
```

### Dự báo volatility

Ưu tiên:

```text
abs_sentiment
sentiment_extremeness
negative_probability
uncertainty
sentiment_dispersion
news_volume
novelty
coverage_breadth
event_severity
event_type
```

---

# 14. Giải thích trường hợp `positive_score` có Pearson r = 0.21 với `log_returns`

Một tương quan Pearson dương khoảng `0.21` giữa `positive_score` và `log_returns` có thể hợp lý về mặt kinh tế:

```text
positive news
→ kỳ vọng tốt hơn
→ lực mua cao hơn
→ return có xu hướng dương
```

Tuy nhiên cần kiểm tra:

```text
1. Correlation có ổn định qua thời gian không?
2. Có còn tồn tại sau khi lag feature không?
3. Có bị tác động bởi một vài outlier không?
4. Có còn tồn tại theo từng cổ phiếu và từng ngành không?
5. Có vượt baseline price-only khi out-of-sample không?
6. Có bị leakage từ thời điểm xuất bản không?
```

Với volatility target, `positive_score` có dấu không nhất thiết là feature mạnh nhất. Nên so sánh:

```text
positive_score
negative_score
abs(sentiment_score)
sentiment_extremeness
uncertainty_score
event_severity
```

Ví dụ một tin cực kỳ tích cực vẫn có thể tạo biến động lớn. Vì thế:

```text
directional feature:
    signed_sentiment

magnitude feature:
    abs(signed_sentiment)
```

---

# 15. Các lỗi cần tránh

## 15.1 Sai về dữ liệu

- Không deduplicate bài đăng lại.
- Không xác định source gốc.
- Gộp tất cả công ty được nhắc vào cùng sentiment.
- Dùng thời điểm crawl thay cho publication time.
- Gắn tin sau giờ đóng cửa vào return cùng ngày.
- Không xử lý cuối tuần và ngày nghỉ.
- Dùng bài báo đã được cập nhật sau thời điểm dự báo.

## 15.2 Sai về feature engineering

- Chỉ dùng một label positive/negative/neutral.
- Dùng trực tiếp 768 embedding dimensions với dataset nhỏ.
- Chỉ lấy sentiment mean, bỏ qua max, std và disagreement.
- Không có relevance score.
- Không có novelty hoặc source breadth.
- Không phân biệt rumor với confirmed event.
- Không normalize event magnitude theo quy mô doanh nghiệp.

## 15.3 Sai về evaluation

- Random train/test split.
- Fit scaler hoặc PCA trên toàn bộ dataset.
- Chọn feature dựa trên toàn bộ thời gian.
- Chỉ nhìn Pearson correlation.
- Không kiểm tra từng horizon riêng.
- Không so với baseline price-only.
- Không kiểm tra statistical significance.
- Không tính transaction cost nếu đánh giá trading strategy.

---

# 16. Quy trình kiểm tra feature hiệu quả thật sự

## Bước 1: Data alignment

```text
news publication timestamp
→ market session mapping
→ ticker mapping
→ trading day aggregation
```

## Bước 2: Baseline

```text
Baseline A: historical volatility only
Baseline B: HAR price-only
Baseline C: OHLCV technical features
```

## Bước 3: Add từng nhóm feature

```text
A + sentiment
A + event
A + novelty
A + dissemination
A + embedding
A + relational
```

## Bước 4: Ablation

```text
Full model
- sentiment
- event
- novelty
- dissemination
- embedding
- relational
```

## Bước 5: Walk-forward validation

Ví dụ:

```text
Train: 2020–2022
Validate: 2023
Test: 2024

Train: 2020–2023
Validate: 2024
Test: 2025
```

Không shuffle dữ liệu thời gian.

## Bước 6: Metrics

### Regression

```text
MAE
RMSE
QLIKE
R² out-of-sample
Pearson correlation
Spearman correlation
```

### Direction/classification

```text
Accuracy
Balanced accuracy
F1
MCC
AUC
```

### Economic evaluation

```text
Sharpe ratio
turnover
maximum drawdown
transaction-cost-adjusted return
```

## Bước 7: Statistical tests

Có thể xem xét:

```text
Diebold–Mariano test
block bootstrap confidence interval
permutation test
feature importance stability
```

---

# 17. Mapping feature vào các model

## 17.1 Ridge

Phù hợp với:

```text
aggregated scalar features
PCA embedding
standardized continuous features
one-hot event categories
```

Cần:

```text
StandardScaler
regularization tuning
walk-forward CV
```

Ridge là baseline tốt để kiểm tra signal tuyến tính.

## 17.2 GBM, XGBoost, LightGBM

Phù hợp với:

```text
nonlinear interactions
missing values
threshold effects
mixed scalar features
```

Đặc biệt hữu ích cho:

```text
novelty × sentiment
uncertainty × current volatility
event severity × liquidity
```

## 17.3 HAR-X

Mở rộng HAR bằng exogenous news variables:

```text
RV_t+1 =
    β0
    + βd × RV_daily
    + βw × RV_weekly
    + βm × RV_monthly
    + γ × NewsFeatures
    + error
```

Có thể tạo:

```text
News_daily
News_weekly
News_monthly
```

tương tự cấu trúc HAR.

## 17.4 GNN

Dùng khi muốn khai thác:

```text
industry spillover
supplier-customer relationship
competitor relationship
co-mention network
dynamic correlation
```

Không nên đưa GNN ngay từ đầu nếu chưa chứng minh được news feature có signal bằng Ridge/GBM/HAR-X.

---

# 18. Schema dữ liệu gợi ý

## 18.1 Article-level table

```text
article_id
source
source_url
publication_timestamp
crawl_timestamp

headline
body
language

target_ticker
target_company
relevance_score
mention_count
actor_is_target
object_is_target

positive_probability
negative_probability
neutral_probability
sentiment_score
sentiment_extremeness
uncertainty_score

event_type
event_subtype
event_status
event_certainty
event_severity

novelty_1d
novelty_7d
duplicate_cluster_id
is_first_report

headline_embedding
body_embedding
```

## 18.2 Event-cluster table

```text
event_cluster_id
first_publication_timestamp
last_publication_timestamp
event_type
target_ticker

article_count
unique_source_count
original_source_count

cluster_sentiment_mean
cluster_uncertainty_mean
cluster_event_severity
coverage_acceleration
```

## 18.3 Daily stock-news table

```text
ticker
trading_date

news_count
unique_source_count

relevance_mean
relevance_max

sentiment_mean
sentiment_max
sentiment_min
sentiment_std
sentiment_extremeness_mean
sentiment_extremeness_max

uncertainty_mean
uncertainty_max

novelty_mean
novelty_max

event_severity_mean
event_severity_max

coverage_breadth
coverage_acceleration

embedding_pca_01
...
embedding_pca_32
```

---

# 19. Danh sách paper nên đọc

## Nhóm nền tảng: textual representation và sentiment

1. **Textual Analysis of Stock Market Prediction Using Breaking Financial News: The AZFin Text System**  
   Robert P. Schumaker, Hsinchun Chen, 2009.  
   DOI: https://doi.org/10.1145/1462198.1462204

2. **A Quantitative Stock Prediction System Based on Financial News**  
   Robert P. Schumaker, Hsinchun Chen, 2009.  
   DOI: https://doi.org/10.1016/j.ipm.2009.05.001

3. **When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks**  
   Tim Loughran, Bill McDonald, 2011.  
   DOI: https://doi.org/10.1111/j.1540-6261.2010.01625.x  
   PDF: https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1540-6261.2010.01625.x

4. **FinBERT: Financial Sentiment Analysis with Pre-trained Language Models**  
   Dogu Araci, 2019.  
   https://arxiv.org/abs/1908.10063

## Nhóm structured event

5. **Using Structured Events to Predict Stock Price Movement: An Empirical Investigation**  
   Xiao Ding et al., EMNLP 2014.  
   https://aclanthology.org/D14-1148/

6. **Incorporating Fine-grained Events in Stock Movement Prediction**  
   Deli Chen et al., 2019.  
   https://arxiv.org/abs/1910.05078

7. **EFSA: Towards Event-Level Financial Sentiment Analysis**  
   Tianyu Chen et al., 2024.  
   https://arxiv.org/abs/2404.08681

## Nhóm novelty, topicality và dissemination

8. **The Impact of the Novelty and Topicality of Business News on Financial Markets**  
   Takayuki Mizuno et al., 2015.  
   https://arxiv.org/abs/1507.06477

9. **Enhancing Sentiment-Based Stock Movement Prediction by Incorporating News Dissemination**  
   2024/2025 preprint.  
   https://arxiv.org/abs/2412.10823

## Nhóm topic modeling

10. **Investor Reaction to Financial Disclosures Across Topics: An Application of Latent Dirichlet Allocation**  
    Stefan Feuerriegel, Nicolas Pröllochs, 2018.  
    https://arxiv.org/abs/1805.03308

11. **Multi-Label Topic Model for Financial Textual Data**  
    Moritz Scherrmann, 2023.  
    https://arxiv.org/abs/2311.07598

## Nhóm contextual embedding và news representation

12. **Stock Movement Prediction with Financial News Using Contextualized Embedding from BERT**  
    Qinkai Chen, 2021.  
    https://arxiv.org/abs/2107.08721

13. **Fine-Tuning Large Language Models for Stock Return Prediction Using Newsflow**  
    Tian Guo, Emmanuel Hauptmann, 2024.  
    https://arxiv.org/abs/2407.18103

## Nhóm temporal news interaction

14. **Stock Movement Prediction from Tweets and Historical Prices**  
    Yumo Xu, Shay B. Cohen, ACL 2018.  
    https://aclanthology.org/P18-1183/

15. **Modeling News Interactions and Influence for Financial Market Prediction (FININ)**  
    Mengyu Wang, Shay B. Cohen, Tiejun Ma, Findings of EMNLP 2024.  
    https://aclanthology.org/2024.findings-emnlp.189/  
    https://arxiv.org/abs/2410.10614

## Nhóm company relations và graph

16. **HATS: A Hierarchical Graph Attention Network for Stock Movement Prediction**  
    Raehyun Kim et al., 2019.  
    https://arxiv.org/abs/1908.07999

17. **Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations**  
    Ramit Sawhney et al., EMNLP 2020.  
    https://aclanthology.org/2020.emnlp-main.676/

18. **Stock Movement Prediction Based on Bi-typed Hybrid-relational Market Knowledge Graph via Dual Attention Networks**  
    Yu Zhao et al., 2022.  
    https://arxiv.org/abs/2201.04965

---

# 20. Thứ tự đọc paper khuyến nghị

Nếu mục tiêu là nhanh chóng chọn feature cho project:

```text
1. Ding et al. 2014
   → structured event

2. Loughran & McDonald 2011
   → financial sentiment và uncertainty

3. Mizuno et al. 2015
   → novelty và topicality

4. Chen et al. 2019
   → fine-grained event

5. FinBERT 2019
   → domain-specific sentiment

6. FT-CE-RNN 2021
   → contextual embeddings

7. FININ 2024
   → news interaction, delayed pricing, long memory

8. EFSA 2024
   → target/event-level sentiment

9. DanSmp hoặc HATS
   → relational và spillover features
```

---

# 21. Khuyến nghị triển khai thực tế cho dữ liệu Việt Nam

## Phase 1: Baseline có khả năng giải thích

```text
relevance
sentiment probabilities
sentiment extremeness
uncertainty
news count
event type
event severity
novelty
unique source count
time decay
```

Model:

```text
Ridge
LightGBM
HAR-X
```

## Phase 2: Semantic features

```text
headline embedding
→ PCA 16/32 dimensions
```

Kiểm tra incremental value so với Phase 1.

## Phase 3: Relational features

```text
sector
competitor
parent-subsidiary
supplier-customer
co-mention
```

Chỉ triển khai GNN sau khi baseline chứng minh được signal.

## Phase 4: Production monitoring

Theo dõi:

```text
feature distribution drift
missing news rate
ticker mapping error
duplicate rate
source concentration
sentiment model drift
event-class frequency drift
out-of-sample feature importance
```

---

# 22. Kết luận cuối

Đối với dự báo volatility từ tin tức, bộ feature có giá trị nhất không phải chỉ là:

```text
positive
negative
neutral
```

Mà là:

```text
Mức liên quan
× Cường độ sentiment
× Bất định
× Loại và độ nghiêm trọng sự kiện
× Độ mới
× Mức độ lan truyền
× Trọng số thời gian
× Trạng thái thị trường
× Quan hệ giữa các công ty
```

Bộ feature nên ưu tiên cho project:

```text
relevance_score
positive_probability
negative_probability
sentiment_score
sentiment_extremeness
uncertainty_score

event_type
event_status
event_certainty
event_severity

novelty_7d
duplicate_cluster_size
unique_source_count
coverage_acceleration

time_decay
recency_weighted_sentiment
recency_weighted_uncertainty
recency_weighted_event_severity

embedding_pca_01 ... embedding_pca_32

sector_sentiment
related_company_sentiment
```

Mọi feature cần được chứng minh bằng:

```text
walk-forward validation
ablation study
price-only baseline comparison
stability across time
stability across stocks
statistical significance
economic usefulness
```
