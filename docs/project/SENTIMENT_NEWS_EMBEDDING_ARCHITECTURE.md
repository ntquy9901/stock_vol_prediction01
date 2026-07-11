# Thiết kế — News Embedding Vector thay cho Sentiment Score

**Ngày:** 05/07/2026
**Tác giả:** ntquy99
**Trạng thái:** Thiết kế + SOTA research, chưa triển khai code
**Liên quan:** `SENTIMENT_ANALYSIS_DESIGN.md` (Phase 3), `SENTIMENT_LATENT_SPACE_TECHNIQUES.md`

---

## 0. Bối cảnh & nguồn gốc

**Thầy góp ý:**

> *"Sentiment score là dữ liệu đã nén mất mát nhiều thông tin. Nếu có thêm embedding vector thì dữ liệu phong phú hơn — các embedding vector của raw tin tức cũng có ý nghĩa nào đó."*

Tài liệu này: (1) giải thích vì sao thầy đúng, (2) research SOTA papers cho hướng embedding, (3) đề xuất kiến trúc mới build trên Parallel LSTM-GNN + Phase 3.

**Quan trọng:** research hiện có trong `_bmad-output/planning-artifacts/research/technical-sentiment-volatility-fusion-sota-2026-06-29.md` đều xử lý sentiment là **scalar** (sentiment_score, news_count). Tài liệu này nâng cấp lên **news embedding vector 768-dim** — hướng riêng, SOTA riêng, đặt ra bài toán kiến trúc mới.

---

## 1. Vì sao thầy đúng — scalar vs embedding

| Khía cạnh | Sentiment **score** (hiện tại) | News **embedding vector** (thầy đề xuất) |
|---|---|---|
| Dung lượng thông tin | 1 scalar [-1, +1] = **nén mất ~99% ngữ nghĩa** | 768-dim vector → giữ **toàn bộ ngữ nghĩa** (entity, sự kiện, mức độ, đối tượng) |
| Ví dụ cùng score=+0.8 | "VCB lợi nhuận tăng 20%" vs "VCB ký hợp đồng lớn" → **giống nhau** | Hai câu ra **2 vector khác hẳn** → model phân biệt được |
| Phụ thuộc NLP head | Chắc chắn (lexicon/PhoBERT classifier) | Dùng embedding pretrained thô → **không cần labeled data** |
| Chi phí | Rẻ | Nặng (768-dim, phải pre-compute offline) |

---

## 2. SOTA papers & methods

| # | Paper | Đóng góp chính | Áp vào project |
|---|---|---|---|
| **1** | **FNSPID** (Dong et al., KDD 2024) — [arXiv 2402.06698](https://arxiv.org/html/2402.06698v1), [GitHub](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) | Benchmark 3 chiến lược: **(a) raw FinBERT embeddings**, **(b) unfreeze FinBERT**, **(c) custom Siamese network**. Dataset 15.7M tin. | **Nền tảng.** Đây chính là "phương pháp" thầy nói. |
| **2** | **Bridging the Gap Between NL and Market Dynamics** — [arXiv](https://arxiv.org/html/2605.30652v1) | Benchmark trên FNSPID: raw embeddings, **attention-weighted aggregation**, Siamese. Kết luận: **Transformer + Siamese embedding > tất cả**. | Xác nhận: aggregation attention là cách đi tốt nhất cho variable news. |
| **3** | **News Sentiment Embeddings for Stock Forecasting** (arXiv 2507.01970, 07/2025) | OpenAI text embeddings trên WSJ headlines → dự báo SPY daily. | Bằng chứng mới nhất 2025: embedding headline ngắn đủ dự báo. |
| **4** | **Multi-Modal Cross Attention Network** (ALTA 2023) — [PDF](https://aclanthology.org/2023.alta-1.7.pdf) | 3 cách embed tin tức + **cross-attention** multimodal. | Pattern fusion price↔news embedding. |
| **5** | **Hierarchical LLM Summary + Co-Attention** (Expert Systems with Applications, 2025) — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417425040564) | Tóm tắt tin bằng LLM rồi co-attention với price. SOTA mới. | Quá phức tạp — defer. |

**Consensus SOTA 2024-2025:** *pre-compute embedding pretrained (FinBERT/PhoBERT/OpenAI) → aggregate variable-length news bằng attention pooling → fuse với price/volatility bằng cross-attention.*

---

## 3. Bài toán kiến trúc MỚI mà embedding đặt ra

Chuyển từ scalar → embedding introduce 3 thách thức mà design hiện tại (Phase 1-3) chưa có:

1. **Variable-length news set**: 1 stock-day có thể có 0, 1, hoặc 15 bài. Phải aggregate theo thứ tự bất biến (permutation-invariant) — **không thể mean như scalar**. → Cần **Attention Pooling / Set Transformer**.
2. **Dimensionality**: 768-dim × 22 days × 30 stocks = nặng. → Bắt buộc **pre-compute offline** (encoder frozen, cache vector), không chạy PhoBERT trong training loop.
3. **"0 news" khó hơn**: với scalar, 0 news = 0; với embedding, 0 news = ??? → cần **learned "no-news" token** + **forward-fill embedding** từ ngày có tin gần nhất.

---

## 4. Kiến trúc đề xuất

So với Phase 3 hiện tại trong `SENTIMENT_ANALYSIS_DESIGN.md`: **thay nhánh sentiment-scalar bằng nhánh news-embedding**, giữ nguyên 2 nhánh HAR (LSTM + GAT) và fusion MSGCA.

```
                    ┌──────────────────────────┬───────────────────────────┬─────────────────────────┐
                    │ TEMPORAL (LSTM) — HAR      │ SPATIAL (GAT) — HAR        │ NEWS EMBEDDING (MỚI)     │
                    │ (giữ nguyên)               │ (giữ nguyên)               │                          │
                    │ input: HAR 22d×3 per stock │ input: HAR, k-NN graph    │ OFFLINE: PhoBERT frozen  │
                    │ → LSTM 2L, h=64            │ → GAT 2L, 4×64=256        │ → title → [768] per bài  │
                    │ → [batch, 30, 64]          │ → [batch, 30, 256]        │                          │
                    │                            │                            │ ONLINE (3 sub-layers):   │
                    │                            │                            │ ① Article-Set Attention  │
                    │                            │                            │   Pooling per (stock,day)│
                    │                            │                            │   → [batch,22,30,d_news] │
                    │                            │                            │ ② News Temporal Encoder  │
                    │                            │                            │   (1-L LSTM over 22 ngày)│
                    │                            │                            │   → [batch,30,d_news]    │
                    │                            │                            │ ③ Projection → d_model   │
                    └──────────┬─────────────────┴────────────┬──────────────┴─────────────┬────────────┘
                               │                              │                            │
                               └──── Q ───────────────────────┴──── K, V ──────────────────┘
                                          GATED CROSS-ATTENTION (MSGCA)
                                          • HAR branch "hỏi" news branch
                                          • gate σ(·): tắt news khi không có tin
                                                  │
                                           MLP → [batch, 30, 1]
```

### 4.1. Component mới quan trọng nhất — Article-Set Attention Pooling

"Trái tim" của chuyển đổi scalar→embedding. Xử lý variable-length news per (stock, day) permutation-invariant, handle "0 news":

```python
class ArticleSetAttentionPooling(nn.Module):
    """
    Aggregate variable-length news embeddings cho 1 (stock, day).
    Input : list of [num_articles, 768]   (0..N bài, N thay đổi mỗi ngày)
    Output: [d_news]                       (1 vector đại diện cả ngày)
    """
    def __init__(self, emb_dim=768, d_news=64):
        super().__init__()
        self.proj = nn.Linear(emb_dim, d_news)
        # learnable "no-news" vector (khi 0 bài trong ngày)
        self.no_news_token = nn.Parameter(torch.randn(d_news) * 0.02)
        # attention query (learnable) — permutation-invariant aggregation
        self.query = nn.Parameter(torch.randn(d_news) * 0.02)

    def forward(self, article_embs, mask):
        # article_embs: [batch, num_stocks, 22, max_articles, 768] (padded)
        # mask        : [batch, num_stocks, 22, max_articles]       (1=bài thật, 0=pad)
        h = self.proj(article_embs)                       # [..., d_news]
        # attention scores: query · each article
        scores = (h * self.query).sum(-1)                 # [..., max_articles]
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)              # [..., max_articles]
        daily = (attn.unsqueeze(-1) * h).sum(-2)          # [..., d_news]
        # nếu 0 bài thật → dùng no_news_token
        has_news = (mask.sum(-1, keepdim=True) > 0).float()
        daily = has_news * daily + (1 - has_news) * self.no_news_token
        return daily
```

→ Chiến lược "attention-weighted aggregation" mà FNSPID/"Bridging the Gap" benchmark thấy thắng mean-pooling.

### 4.2. Offline news embedding pipeline (data, không phải model)

Thêm 1 bước vào pipeline 6 bước hiện tại (chèn giữa bước 3 và 4):

```
[3] Score sentiment (giữ)  ──►  [3b] EMBEDDING EXTRACTION (MỚI)
                                   - PhoBERT frozen, title → CLS [768]
                                   - Cache: data/sentiment_baseline/{TICKER}_embeddings.npz
                                   - Key = date, Value = [num_articles, 768]
[4] Aggregate scalar ──► [4b] Aggregate scalar (giữ) + giữ nguyên list embedding
```

**Tại sao frozen + offline:** train PhoBERT end-to-end với ~12K title tiếng Việt → overfit chắc chắn (FNSPID cũng unfreeze chỉ khi có 15M tin). Pre-compute 1 lần, training chỉ dùng vector cache → nhanh như Phase 1.

### 4.3. Dataset class đổi gì

`MultiStockDatasetWithSentiment` hiện trả 5 feature scalar. Phiên bản embedding cần trả **thêm**:
- `x_emb`: `[22, 30, max_articles, 768]` + `mask` `[22, 30, max_articles]`
- Giữ `x_scalar` (5 feat) để model dùng kèm (embedding + scalar complementary)

→ `__getitem__` trả tuple `(x_har, adj, x_emb, mask, y)` thay vì `(x, adj, y)`.

---

## 5. Đánh giá trung thực + phasing

| Nhận định | Chi tiết |
|---|---|
| ✅ Hướng đúng | Đồng nhất với SOTA 2024-2025 (FNSPID, Bridging the Gap). Cite mạnh cho báo cáo thầy. |
| ⚠️ Bottleneck vẫn là data | Title-only tiếng Việt ~12K bài. Embedding 768-dim từ title ngắn VN **vẫn nhiễu** (chưa có body). Embedding giàu hơn score nhưng **không tạo tín hiệu không có**. |
| ⚠️ Nặng hơn nhiều | 768-dim → RAM/VRAM tăng, phải pre-compute offline. 0-news problem phức tạp hơn. |
| 🔴 Vẫn cần tin test | Thiếu tin test 2021-2025 → embedding cũng không lift test được (Mục 10 design). |
| 📐 Phức tạp vs Simplicity | Embedding branch + attention pooling + cross-attn = **~200 dòng mới**. Chỉ làm nếu Phase 1-2 đã có tín hiệu. |

### Phasing đề xuất (verify từng bước trước khi bước tiếp)

```
Bước 0 (đANG): Phase 1 scalar (đã build) + crawl thêm tin test 2021-2025  [verify: tin test đủ]
Bước 1: Phase 1 scalar + scorer PhoBERT (đã code, chưa run)              [verify: scalar có tín hiệu?]
Bước 2 (MỚI): Offline embedding PhoBERT + Article-Set Attention Pooling
              + late-fusion concat (KHÔNG cross-attn)                     [verify: embedding > scalar trên val?]
Bước 3 (MỚI): Gated cross-attention (MSGCA) fusion                        [verify: lift thêm?]
Bước 4: (tùy chọn) unfreeze PhoBERT head / Siamese (FNSPID)               [chỉ nếu data đủ lớn]
```

> **Quyết định go/no-go cốt lõi ở Bước 2:** so sánh val Dir Acc của (scalar sentiment) vs (embedding sentiment). Nếu embedding **không hơn** scalar trên val → tín hiệu gốc yếu, dừng, embedding chỉ tốn compute. Đừng đầu tư Bước 3-4.

---

## 6. Sources

- [FNSPID (KDD 2024, arXiv 2402.06698)](https://arxiv.org/html/2402.06698v1)
- [FNSPID GitHub](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)
- [Bridging the Gap: NL & Market Dynamics](https://arxiv.org/html/2605.30652v1)
- [News Sentiment Embeddings for Stock Forecasting (arXiv 2507.01970)](https://arxiv.org/html/2507.01970v1)
- [Multi-Modal Cross Attention Network (ALTA 2023)](https://aclanthology.org/2023.alta-1.7.pdf)
- [Hierarchical LLM Summary + Co-Attention (ESWA 2025)](https://www.sciencedirect.com/science/article/abs/pii/S0957417425040564)

---

**Phiên bản:** 1.0 — 05/07/2026
