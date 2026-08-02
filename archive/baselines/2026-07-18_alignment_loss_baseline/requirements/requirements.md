# Requirements (Specify) — M2VN-style Alignment-Loss Baseline

**Baseline:** `2026-07-18_alignment_loss_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** M2VN — "Fusing Narrative Semantics for Financial Volatility Forecasting"
(Kong et al., Oxford/ICAIF'25, arXiv:2510.20699) — xem
`_bmad-output/planning-artifacts/research/technical-sparse-news-volatility-forecasting-sota-research-2026-07-18.md`.

## 1. Ý tưởng

M2VN dùng **auxiliary alignment loss** để kéo latent space của tin tức (unstructured) và giá
(structured) lại gần nhau — mục đích: ép nhánh news học representation "tương thích" với không
gian mà nhánh HAR đã dùng để dự báo tốt, thay vì học một representation độc lập/không liên quan
(chính là biểu hiện khác của "text collapse").

**Không tái tạo được:** M2VN dùng Time Machine GPT (point-in-time LLM) — không có model này,
bỏ qua phần đó. Chỉ implement phần **alignment loss**, phần khả thi và độc lập.

## 2. Cơ chế

- 2 projection head chiếu `har_embed = concat[h_lstm,h_gnn]` (320-d) và `news_rep` (64-d) về
  CÙNG 1 không gian chung `d_align` (vd 32-d).
- Auxiliary loss: `1 - cosine_similarity(proj_har, proj_news)` (align loss), CỘNG vào loss
  chính (MSE dự báo, dùng concat fusion như `EmbeddingBaseline` gốc — không đổi kiến trúc dự
  báo, chỉ thêm loss phụ).
- `loss = MSE(pred, y) + lambda_align * align_loss` (lambda mặc định 0.1).

## 3. Data — TÁI DÙNG

`data/sentiment_embedding/{TICKER}_emb.npz` (giống REST-TS baseline, để so sánh công bằng).

## 4. Training policy

15 epoch (user đã duyệt tối 2026-07-18 cho cả 3 baseline mới), learning curve mỗi 5 epoch.
