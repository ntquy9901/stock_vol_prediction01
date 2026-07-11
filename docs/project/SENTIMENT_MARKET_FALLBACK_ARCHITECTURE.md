# Thiết kế — Market-Level Sentiment Fallback

**Ngày:** 08/07/2026
**Trạng thái:** Design proposal (chưa implement)
**Nguồn gốc:** Khuyến nghị #3 trong `EMBEDDING_BASELINE_REPORT_2026-07-08.md` (xử lý no-lift do sparsity)
**Tham chiếu:** `SENTIMENT_NEWS_EMBEDDING_ARCHITECTURE.md`, sparsity report `crawl_data/aggregated/sparsity_report.txt`, `_bmad-output/planning-artifacts/sentiment-sparsity-solution-2026-06-29.md` (Solution 3)

---

## 0. TL;DR

Thêm 1 nhánh **market news** (dense, ~100% ngày có) bên cạnh nhánh **per-stock news** (sparse, 5.5%). Dùng **gate** chọn stock-specific khi có, fallback sang market khi stock mù tin. Reuse `ArticleSetAttentionPooling` + PhoBERT cache đã có (chỉ bỏ ticker filter khi extract). ~60-90 dòng code mới, không đụng HAR branch.

---

## 1. Vấn đề giải quyết

Từ sparsity analysis:
- **~80% bài là vĩ mô** (không match mã VN30) → đang bị `extract_embeddings.py` bỏ.
- **94.5% stock-day ở test** không có tin ticker-specific → nhánh news embedding hiện = `no_news_token` hầu hết lúc.
- Nhưng tin vĩ mô **ảnh hưởng mọi cổ phiếu** (systematic mood) → đang lãng phí.

→ Market fallback biến bài toán "per-stock thưa 5.5%" thành "per-day dense ~100%" ở cấp market.

## 2. Kiến trúc tổng thể

```
                        ┌──────────────────────────────────────────────────┐
                        │  HAR BRANCH (giữ nguyên — LSTM + GAT)             │
                        └────────────────┬─────────────────────────────────┘
                                         │ h_lstm, h_gnn  [B, stocks, 64/256]
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        │                                                                │
        ▼                                                                ▼
┌───────────────────────────┐                          ┌────────────────────────────────┐
│ PER-STOCK NEWS (hiện tại)  │                          │ MARKET NEWS (MỚI — dense)       │
│ x_emb[B,seq,stocks,MAX,dim]│                          │ x_market[B,seq,MAX_M,dim]       │
│ + mask                     │                          │ + market_mask                   │
│ → ArticleSetAttentionPool  │                          │ → ArticleSetAttentionPool       │
│   → stock_daily            │                          │   → market_daily [B,seq,d_news] │
│   [B,seq,stocks,d_news]    │                          │   (1 vector/ngày, shared)       │
│   (0-news day=no_news_tok) │                          │   (luôn có giá trị — dense)     │
└─────────────┬─────────────┘                          └───────────────┬────────────────┘
              │                                                        │
              │  has_news = (mask.sum(-1)>0)  [B,seq,stocks,1]         │ broadcast→stocks
              └──────────────────┐    ┌────────────────────────────────┘
                                 ▼    ▼
                          ┌─────────────────────────────────────┐
                          │  GATED NEWS FUSION                   │
                          │  g = has_news  (MVP deterministic)   │
                          │  daily = g·stock + (1−g)·market      │
                          │  → [B,seq,stocks,d_news]             │
                          └────────────────┬─────────────────────┘
                                           │
                                           ▼
                          ┌─────────────────────────────────────┐
                          │  News Temporal LSTM (giữ nguyên)     │
                          │  → news_rep [B,stocks,d_news]        │
                          └────────────────┬─────────────────────┘
                                           │
                                           ▼
                          concat[h_lstm, h_gnn, news_rep] → MLP → pred [B,stocks]
```

## 3. Gate design

**MVP (recommend) — deterministic, availability-based:**
```python
g = has_news.float()   # 1 nếu stock CÓ tin ngày đó, 0 nếu KHÔNG
daily = g * stock_daily + (1 - g) * market_daily_broadcast
```
- Có tin stock → dùng stock-specific (đáng tin nhất).
- Không tin stock → fallback market (dense, không bao giờ = no_news_token).
- 0 tham số học, interpret trực tiếp, không risk gate học sai.

**Option (refinement) — learned soft gate:**
```python
# g ∈ (0,1) mềm, học được từ [stock_emb, market_emb, news_count]
g = sigmoid(MLP(concat[stock_daily, market_daily, has_news]))
daily = g * stock_daily + (1 - g) * market_daily
```
- Cho phép blend (vd 70% stock + 30% market) thay vì binary.
- Risk: gate có thể học "luôn dùng HAR" → triệt tiêu nhánh news. **Chỉ làm nếu MVP có tín hiệu.**

## 4. Component mới — pseudocode

### 4.1. `MarketBranch` (reuse `ArticleSetAttentionPooling`)
```python
class MarketBranch(nn.Module):
    """Pool TẤT cả bài trong 1 ngày → 1 market vector/day. Reuse ArticleSetAttentionPooling."""
    def __init__(self, emb_dim: int, d_news: int):
        super().__init__()
        self.pool = ArticleSetAttentionPooling(emb_dim, d_news)  # cùng module đã có!

    def forward(self, x_market, market_mask):
        # x_market: [B, seq, MAX_M, emb_dim]   (tất cả bài ngày đó, KHÔNG lọc ticker)
        # market_mask: [B, seq, MAX_M]
        return self.pool(x_market, market_mask)   # [B, seq, d_news]
```

### 4.2. `GatedNewsFusion`
```python
class GatedNewsFusion(nn.Module):
    """Fuse per-stock news (sparse) + market news (dense) theo availability."""
    def __init__(self, d_news: int, learned_gate: bool = False):
        super().__init__()
        self.learned = learned_gate
        if learned_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_news * 2 + 1, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, stock_daily, market_daily, has_news):
        # stock_daily: [B, seq, stocks, d_news]
        # market_daily: [B, seq, d_news]
        # has_news: [B, seq, stocks, 1]
        market = market_daily.unsqueeze(2).expand_as(stock_daily)  # broadcast → stocks
        if self.learned:
            g = self.gate_mlp(torch.cat([stock_daily, market, has_news], dim=-1))
        else:
            g = has_news   # deterministic MVP
        return g * stock_daily + (1 - g) * market     # [B, seq, stocks, d_news]
```

### 4.3. Tích hợp vào `EmbeddingBaseline.forward`
```python
def forward(self, x_har, adj, x_emb, mask, x_market, market_mask):
    # HAR branch (giữ nguyên)
    h_lstm, h_gnn = self.har.get_embeddings(x_har, adj)

    # Per-stock news (giữ nguyên)
    stock_daily = self.news_pool(x_emb, mask)                  # [B,seq,stocks,d_news]

    # MỚI: market news + gated fusion
    market_daily = self.market_branch(x_market, market_mask)   # [B,seq,d_news]
    has_news = (mask.sum(-1, keepdim=True) > 0).float()        # [B,seq,stocks,1]
    daily = self.gated_fusion(stock_daily, market_daily, has_news)  # [B,seq,stocks,d_news]

    # Temporal (giữ nguyên)
    news_rep = self.news_temporal(daily)                       # [B,stocks,d_news]

    h = torch.cat([h_lstm, h_gnn, news_rep], dim=-1)
    return self.fusion(h).squeeze(-1)
```

## 5. Cache format — market embedding

**File mới:** `data/sentiment_embedding/market_emb.npz`
- Keys: date string `YYYY-MM-DD` (mọi ngày có ≥1 bài, bất kể ticker)
- Values: `[n_articles_that_day, dim]` (dim=64, cùng PCA đã fit)
- **~17K bài vĩ mô** (80% của 21K) → ~17K array rows trên ~2500 ngày giao dịch → trung bình ~7 bài/ngày.

**Cách extract (2 lựa chọn):**

| Cách | Ưu | Nhược |
|---|---|---|
| **A. Flag `--market` trong `extract_embeddings.py`** | reuse pipeline | thêm nhánh logic |
| **B. Script riêng `extract_market_embeddings.py`** | cô lập rõ | duplicate ~30 dòng |

→ **Recommend A:** thêm `--market` (bỏ ticker filter, group theo date thay vì ticker). Code diff:
```python
if args.market:
    # bỏ pat.findall() filter — embed TẤT cả bài
    # group theo date (không theo ticker) → cache market_emb.npz
```

## 6. Dataset changes — `dataset_embedding.py`

```python
class MultiStockDatasetWithEmbedding(...):
    def __init__(self, *args, market_cache_path=None, **kwargs):
        ...
        self._market = self._load_market(market_cache_path)  # {date: [n,dim]}

    def _create_sequences(self):
        ...
        for i in range(...):
            window_dates = [_norm_date(d) for d in ...]
            # per-stock x_emb (hiện tại — giữ)
            ...
            # MỚI: market embedding per day (shared all stocks)
            market_day_embs = [self._pad_articles(self._market.get(d))[0]
                               for d in window_dates]   # [seq, MAX_M, dim]
            market_day_masks = [self._pad_articles(self._market.get(d))[1]
                                for d in window_dates]  # [seq, MAX_M]
            x_market = np.stack(market_day_embs)   # [seq, MAX_M, dim]
            market_mask = np.stack(market_day_masks)
            sequences.append((x_har, adj, x_emb, mask, x_market, market_mask, y))

    def __getitem__(self, idx):
        # trả thêm x_market, market_mask (không normalize — model proj xử lý)
        return (x_har, adj, x_emb, mask, x_market, market_mask, y)
```

**Lưu ý:** `MAX_M` (market cap/ngày) có thể lớn hơn `MAX_ARTICLES` per-stock (ngày đông tin vĩ mô). Đặt `MAX_M=15` hoặc compute percentile. x_market **không duplicate per-stock** (1 vector/ngày broadcast trong model) → tiết kiệm RAM.

## 7. Hyperparams & config

| Param | Giá trị | Lý do |
|---|---|---|
| `d_news` | 64 | ngang per-stock branch |
| `MAX_M` (market articles/day) | 15 | percentile 99 bài/ngày |
| `learned_gate` | False (MVP) | đơn giản, không risk; True chỉ nếu MVP có tín hiệu |
| `gap_threshold` (learning curve) | 0.05 | giữ nguyên |

Train: reuse toàn bộ train script của embedding baseline (chỉ đổi forward signature + dataloader).

## 8. Go/no-go criteria

| # | Tiêu chí | Verify |
|---|---|---|
| 1 | Pipeline chạy end-to-end (extract market → dataset → train) không lỗi | smoke test pass |
| 2 | Market coverage > 90% ngày (gần mọi stock-day có market signal) | check `market_emb.npz` date coverage |
| 3 | **Go/no-go cốt lõi:** val DirAcc **market-fallback > embedding-baseline (hiện tại)** tại matched epoch | so results.json |
| 4 | Gate không sụp (var(news_rep) > 0, không collapse về HAR) | check learning curve + news branch activation |

**NO-GO nếu:** market fallback không vượt embedding baseline hiện tại → market signal cũng không đủ (xác nhận ceiling do data, không phải architecture).

## 9. Đánh giá trung thực

| | |
|---|---|
| ✅ Giải đúng root cause | dense coverage thay vì 5.5% |
| ✅ Reuse | `ArticleSetAttentionPooling` + PhoBERT cache (bỏ ticker filter) |
| ✅ Surgical | không đụng HAR branch, ~60-90 dòng mới |
| ⚠️ Ceiling | market = systematic mood, không stock-specific alpha. Lift kỳ vọng **nhỏ** (giúp ngày toàn thị trường biến động mạnh) |
| ⚠️ Confound | vẫn cần matched-epoch control vs embedding baseline hiện tại |
| 📐 Quyết định | chỉ implement nếu matched-epoch control của embedding baseline xong VÀ vẫn no-lift (market fallback là bước tiếp theo hợp lý) |

## 10. Implementation plan (nếu go)

```
baselines/2026-07-XX_market_fallback/   (rule §3.F: 5 sub-folder)
├── requirements/   ← success criteria mục 8
├── design/         ← copy doc này
├── code/
│   ├── extract_embeddings.py  (copy + thêm --market flag)
│   ├── dataset_embedding.py   (copy + thêm market cache loading)
│   ├── model_embedding.py     (copy + thêm MarketBranch + GatedNewsFusion)
│   └── train_*.py             (copy, đổi forward unpacking)
├── code_review/    ← adversarial review trước khi "done"
└── test/           ← test gate correctness + market branch shape/permutation
```

Estimate: ~1 ngày implement + review + matched-epoch comparison.

---

**Phiên bản:** 1.0 — 08/07/2026 · **Recommend:** implement SAU khi matched-epoch control của embedding baseline xong và vẫn no-lift.
