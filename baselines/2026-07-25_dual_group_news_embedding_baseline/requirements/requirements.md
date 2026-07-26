# Requirements (Specify) — Dual-Group News Embedding Baseline

**Baseline:** `2026-07-25_dual_group_news_embedding_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** `C:\luanvan\data_eda\docs\embedding_pipeline_reference.md` (pipeline PhoBERT →
PCA → dual-group EWMA của project `data_eda`).

## 1. Vấn đề đang giải quyết

6 baseline news hiện có trong `stock_vol_prediction01` (embedding, latent-noise, market-fallback,
objective-news, pure-market, REST-TS/alignment-loss/gated-crossattn) đều dùng **1 nguồn embedding
duy nhất**: PhoBERT → PCA-64, KHÔNG phân biệt loại nguồn tin (`data/sentiment_embedding/{TICKER}_emb.npz`).

Project sibling `data_eda` (`C:\luanvan\data_eda`) đã xây một pipeline embedding **phong phú hơn**,
CHƯA từng dùng trong `stock_vol_prediction01`:
- Tách 2 nhóm nguồn tin loại trừ lẫn nhau: `khach_quan` (báo chí khách quan) vs `tong_hop`
  (bình luận/phân tích của CTCK) — 2 tín hiệu có thể mang thông tin khác nhau.
- Multi-window EWMA (5/10/20/30/60 ngày), novelty, dispersion, max semantic shock — nắm bắt
  "tin tức mới bất thường" thay vì chỉ trung bình embedding trong ngày.

Baseline này trả lời: **dùng bộ feature dual-group embedding đã có sẵn từ `data_eda` có cải thiện
Dir Acc so với baseline PCA-64 gốc không?**

## 2. Nguồn dữ liệu — TÁI DÙNG, KHÔNG rebuild PhoBERT

**Ràng buộc cứng (user 2026-07-24):**
- KHÔNG chạy lại bước PhoBERT extraction (đã cache, tốn ~8min→nhiều giờ nếu miss cache).
- KHÔNG sửa bất kỳ file nào trong `C:\luanvan\data_eda` (project của người khác/session khác).
- Bất kỳ file (code hoặc data) nào cần dùng từ `data_eda` → **copy sang `stock_vol_prediction01`
  trước**, làm việc trên bản copy, không đọc/ghi trực tiếp vào `data_eda`.

**Phát hiện staleness (kiểm tra timestamp 2026-07-24):** `data_eda/eda_output/modeling/
advanced_news_features.parquet` (panel đã build sẵn) build lúc 2026-07-22 — CŨ hơn ~20 nguồn tin
mới backfill lúc 2026-07-23/24 (baodautu, cafebiz, coin68, fica, forum, nhadautu,
nhipsongkinhdoanh, telegram_*, theinvestor, thoibaotaichinhvietnam, thuonghieucongluan,
tinnhanhchungkhoan, vietbao, vietnambiz, vietnamfinance, vietnamnet, vneconomy, vnstock, + biến
thể `*_root`). Dùng panel có sẵn as-is sẽ THIẾU tín hiệu tin tức 2 ngày gần nhất.
→ **Quyết định (user xác nhận):** copy per-source PhoBERT cache (`data/features/
news_emb_articles_{source}.parquet`, đã fresh tính đến 2026-07-24 21:49) + code aggregation liên
quan sang project này, tự chạy lại bước aggregation (PCA/EWMA/novelty/dispersion — KHÔNG phải
PhoBERT) để có panel fresh. Vì mọi article's `url` đã có trong cache, bước này sẽ KHÔNG gọi
PhoBERT (0 cache-miss dự kiến) — chỉ là pandas/PCA, ~vài phút.

## 3. Scope — Feature set (Simplicity Gate)

`embedding_pipeline_reference.md` §3.5 liệt kê 3 mức: `ADV_FEATURES_DUAL` (80 cột: basic dual +
topic), `+ EWMA_FEATURES` (66 cột, single 30d), `+ EWMA_MULTI/novelty/dispersion/shock` (full,
~480 cột). Theo CLAUDE.md §2 Simplicity First — bắt đầu với **basic + single-EWMA (146 cột)**
thay vì full 480 cột ngay: đủ để kiểm tra tín hiệu dual-group có ích không, tránh train một model
lớn/chậm chưa biết có "text collapse" hay không (bài học từ REST-TS baseline). Nếu basic+EWMA cho
kết quả khả quan → follow-up baseline mở rộng multi-EWMA+novelty+dispersion.

**[NEEDS CLARIFICATION - đã quyết định mặc định, có thể điều chỉnh]:** dùng `mode="ewma"` của
`build_advanced_features()` (basic dual + single 30d EWMA), KHÔNG dùng `mode="full"`.

## 4. Kiến trúc — TÁI DÙNG pattern `2026-07-07_embedding_baseline`

Theo pattern đã chọn (user xác nhận): giữ nguyên `ParallelLSTMGNN.get_embeddings` (HAR branch,
đọc-only) + thay nhánh news CŨ (PCA-64 đơn, article-set attention pooling) bằng nhánh news MỚI
tiêu thụ **vector đã aggregate sẵn theo (ticker, ngày)** — ĐƠN GIẢN HƠN bản gốc vì không cần
pad/mask article-set (mỗi ngày đã là 1 vector cố định chiều, không phải tập hợp N bài báo).

```
forward:
  h_lstm, h_gnn = ParallelLSTMGNN.get_embeddings(x_har, adj)     # [B,S,64], [B,S,256]  (như cũ)
  news_rep      = NewsFeatureLSTM(x_news)                        # [B,S,d_news]  (LSTM 1 lớp qua 22 ngày)
  h = concat([h_lstm, h_gnn, news_rep])
  pred = MLP(h)
```

So sánh trực tiếp với `EmbeddingBaseline` gốc (68.76%@40ep, hoặc 68.44%/70.29% @5ep) — cùng
HAR branch, cùng split, chỉ khác nguồn+kiến trúc nhánh news.

## 5. Success criteria (go/no-go)

- **Go:** pipeline chạy hết không lỗi, tạo được panel fresh (ticker, date, 146 cột dual+EWMA),
  toàn bộ 6 metric bắt buộc (MSE/RMSE/MAE/R²/QLIKE/DirAcc) tính được cho val+test, model train
  ổn định (loss giảm, không NaN) trong 10 epoch thử nghiệm (Training policy CLAUDE.md).
- **So sánh:** DirAcc của baseline mới vs 68.76% (EmbeddingBaseline gốc, 40ep) / vs
  68.44%-70.29% (5ep) — ghi nhận kết quả dù thắng hay thua, không cần vượt để coi "done" (đây là
  thí nghiệm SOTA-comparison, không phải production gate).
- **No-go / dừng sớm nếu:** panel fresh build ra toàn NaN (bug ticker-explode/date-match), hoặc
  coverage (số ngày có tin tức thực) < mức của baseline gốc → nghi ngờ bug trong bước vendor/copy,
  cần fix trước khi train.

## 6. Training policy

10 epoch thử nghiệm đầu (CLAUDE.md Training policy) — báo cáo val metrics sau mỗi 5 epoch +
learning curve, KHÔNG tự ý vượt 10 epoch khi chưa có xác nhận từ user.
