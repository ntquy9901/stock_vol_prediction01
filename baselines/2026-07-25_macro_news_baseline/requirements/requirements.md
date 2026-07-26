# Requirements (Specify) — Market-Wide Macro News Baseline

**Baseline:** `2026-07-25_macro_news_baseline` · Theo SDD (CLAUDE.md §1.5).
**Depends on:** `2026-07-25_expand_news_cache_baseline` (`--include_all` run) — needs
`data/external_news_embeddings/raw_cache/` to contain embeddings for articles that do NOT
mention any VN30 ticker (the whole point of this baseline).

## 1. Bối cảnh

User yêu cầu: sau khi mở rộng cache PhoBERT sang TOÀN BỘ bài viết (kể cả bài không nhắc ticker
nào, dùng GPU, ~7.49 triệu bài), hãy tạo 1 baseline THỬ dùng embedding này. Cơ chế hiện có
(`dual_group_news_panel.parquet`) chỉ dùng bài CÓ nhắc ticker, tổng hợp theo (ticker, date). Bài
KHÔNG nhắc ticker nào (tin vĩ mô/thị trường chung — chính sách tiền tệ, lãi suất, VN-Index nói
chung, tin ngành không gắn 1 mã cụ thể) trước đây bị loại bỏ hoàn toàn.

## 2. Giả thuyết

Tin vĩ mô/thị trường chung (không gắn ticker) mang tín hiệu HỮU ÍCH cho biến động của TẤT CẢ 30
mã (không phải tín hiệu riêng cho từng mã) — nên tổng hợp theo NGÀY (không theo ticker), rồi
BROADCAST cùng 1 vector cho cả 30 mã tại ngày đó, cộng thêm vào feature vector hiện có
(dual-group, 146 cột) của từng mã.

## 3. Mục tiêu / Output

1. `build_macro_panel.py` → `data/features/macro_news_panel.parquet`: (date, macro_emb_0..31,
   ewma_macro_emb_0..31, macro_emb_norm) — tổng hợp TOÀN BỘ bài viết (mọi nguồn, không lọc
   ticker) theo ngày giao dịch hiệu lực (effective_trading_date, tái dùng
   `phase04_news_helpers`).
2. Dataset mới (`dataset_macro_news.py`) nối macro panel (date-only, broadcast) vào dual-group
   panel hiện có (per-ticker) → x_news rộng hơn = 146 + macro_dims.
3. Model: **tái dùng `DualGroupNewsBaseline`/`build_default_model` KHÔNG đổi** (đã n_feat-agnostic,
   xem `model_dual_news.py` — chỉ cần n_feat lớn hơn).
4. Train script mirror `train_dual_news.py`, 10 epoch (theo Training policy — user đi ngủ, không
   thể xin approve >10 epoch tối nay, giữ cap mặc định).

## 4. Cô lập (hard isolation, CLAUDE.md §3.F rule 3)

Import read-only từ `2026-07-25_dual_group_news_embedding_baseline` (`model_dual_news.py`,
`dataset_dual_news.load_news_panel`, `vendor_data_eda.phase04_news_helpers`,
`vendor_data_eda.discover_news`, `vendor_data_eda.news_embeddings.{_article_cache_path,RAW_DIM,
TRAIN_CUTOFF}`) và từ `2026-07-25_expand_news_cache_baseline` (không cần — panel build tự đọc
raw_cache trực tiếp). KHÔNG sửa file nào của 2 baseline đó.

## 5. PCA leakage safety

Tái dùng `TRAIN_CUTOFF="2010-06-30"` (đã derive đúng cho split hiện tại của project, xem
`[[feedback_cross_project_vendoring]]`) — fit PCA(32) trên bài viết TRƯỚC cutoff này (toàn bộ
nguồn, không lọc ticker). Honest fallback (giữ full 768-dim nếu quá ít mẫu pre-cutoff) — pattern
giống `news_embeddings.py::_reduce`.

## 6. Success criteria / Go-No-go

- [ ] `macro_news_panel.parquet` build được, không NaN toàn bộ, có ~4890 ngày (khớp trading
      calendar hiện tại).
- [ ] Dataset shapes đúng: x_news width = 146 + macro_dims.
- [ ] Model forward + backward không lỗi (smoke test).
- [ ] Train 10 epoch thật, in đủ 6 metrics mỗi 5 epoch + val/test comparison.
- [ ] So sánh với dual-group baseline hiện có (68.50% test DirAcc, R²=0.7157 [gated-crossattn
      record]) — không kỳ vọng thắng, đây là baseline THỬ (per user: "thử tiếp").
- [ ] pytest pass, code review chạy trước khi coi "done".

## 7. Out of scope

- Không tune hyperparameter sâu (đây là lần thử đầu).
- Không thêm attention/gating riêng cho macro feature — concat đơn giản trước, đúng Simplicity
  Gate; nếu có tín hiệu mới xét gating (giống gated_crossattn) sau.
