# Requirements (Specify) — MSGCA-style Gated Cross-Attention Baseline

**Baseline:** `2026-07-18_gated_crossattn_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** MSGCA — "Stock movement prediction with multimodal stable fusion via gated
cross-attention mechanism" (Complex & Intelligent Systems, 2025) — xem
`_bmad-output/planning-artifacts/research/technical-sparse-news-volatility-forecasting-sota-research-2026-07-18.md`.

## 1. Ý tưởng

Tất cả baseline trước dùng **concat + MLP** (fusion tĩnh, không có cơ chế lựa chọn) hoặc
**gate nhị phân cố định** (`has_news`, ở market_fallback). MSGCA dùng **gated cross-attention
học được**: modality chính (HAR/giá) làm "query" attend vào modality phụ (news) qua
cross-attention, rồi 1 gate (sigmoid, HỌC ĐƯỢC — không cố định) quyết định pha trộn bao nhiêu
tín hiệu news vào representation cuối.

## 2. Cơ chế

- `har_embed` (query) attend vào `news_rep` (key/value) qua 1 lớp multi-head cross-attention →
  `attended_news`.
- Gate học được: `g = sigmoid(Linear(concat[har_embed, attended_news]))` (giống tinh thần MSGCA
  — gate phụ thuộc CẢ 2 modality, không phải chỉ `has_news` nhị phân).
- `fused = har_embed + g * attended_news` (residual-gated, ổn định hơn concat cứng).
- `pred = MLP(fused)`.

## 3. Data — TÁI DÙNG

`data/sentiment_embedding/{TICKER}_emb.npz` (giống 2 baseline kia, so sánh công bằng).

## 4. Training policy

15 epoch (đã duyệt), learning curve mỗi 5 epoch.
