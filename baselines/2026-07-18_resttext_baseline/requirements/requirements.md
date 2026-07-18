# Requirements (Specify) — REST-TS Residual-Supervision Baseline

**Baseline:** `2026-07-18_resttext_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** `_bmad-output/planning-artifacts/research/technical-sparse-news-volatility-forecasting-sota-research-2026-07-18.md`
— "Does Text Actually Help? Uncovering and Resolving Text Collapse in Multimodal Time Series
Forecasting" (Nguyen et al., Deakin, arXiv:2606.19413, 2026).

## 1. Vấn đề đang giải quyết

Cả 5 baseline news trước (embedding, latent-noise, market-fallback, objective-news, pure-market)
đều dùng concat + 1 loss chung — nhánh HAR (autocorrelated mạnh với target) áp đảo optimization,
nhánh news không bị BẮT BUỘC phải học gì hữu ích ("text collapse", theo đúng tên gọi trong paper).

## 2. Ý tưởng REST-TS

Tách nhánh HAR và nhánh news thành 2 "đầu" dự báo ĐỘC LẬP:
- `har_pred` = MLP trên riêng h_lstm+h_gnn, train bằng loss chính (MSE với y).
- `news_pred` = MLP trên riêng news_rep, train bằng loss PHỤ trên **residual**
  `(y - har_pred.detach())` — phần sai số mà HAR KHÔNG giải thích được.
- Vì không có đường nào từ numerical pathway giảm được residual loss này, nhánh news bị ép phải
  học tín hiệu thật (nếu không sẽ luôn predict residual≈0, loss phụ vẫn cao).
- Dự báo cuối = `har_pred + news_pred`.

## 3. Data — TÁI DÙNG

`data/sentiment_embedding/{TICKER}_emb.npz` (embedding baseline 2026-07-07, ticker-matched broker
reports) — cùng data với baseline gốc 68.76%@40ep để so sánh công bằng CHỈ khác cơ chế fusion/loss.

## 4. Success criteria

So `har_pred+news_pred` (REST-TS) vs `EmbeddingBaseline` gốc (concat+MLP, cùng data, 10 epoch) —
kỳ vọng REST-TS giảm/loại bỏ collapse, cải thiện DirAcc so với 68.44%/70.29% (5ep) hoặc
68.76%(40ep) đã có.

## 5. Training policy

10 epoch (thí nghiệm mới, theo CLAUDE.md Training policy — không tự ý vượt quá khi chưa có
review từ user, dù user đã cho phép tự chạy tối nay).
