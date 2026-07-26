# Requirements (Specify) — Expand News-Embedding Cache

**Baseline:** `2026-07-25_expand_news_cache_baseline` · Theo SDD (CLAUDE.md §1.5).

## 1. Bối cảnh

User đã crawl thêm dữ liệu tin tức vào `C:\luanvan\crawl_data\data` (sibling crawler, không phải
data_eda). Scoping (2026-07-25) cho thấy:
- **12 nguồn HOÀN TOÀN MỚI**, chưa có cache nào: `baophapluat`, `bnews`, `cand`, `dantri`,
  `giaoducthoidai`, `hanoimoi`, `plo`, `sggp`, `tapchicongthuong`, `tienphong`, `viettimes`, `vov`.
  Tất cả đều là báo chí phổ thông (mainstream press) — cùng loại với các nguồn `khach_quan` hiện
  có (cafef, vnexpress, thanhnien, tuoitre, dantri-style...).
- **~10 nguồn đã có cache nhưng crawl đã thêm bài mới** (cafef +7, nhipsongkinhdoanh +352,
  theinvestor +31, v.v.) — cần bổ sung phần bài mới, KHÔNG encode lại toàn bộ.
- **Tổng: 13,818 bài viết có nhắc VN30 ticker, chưa có trong cache** (đo bằng script scoping
  read-only, không gọi PhoBERT). Ước tính encode: ~9-15 phút trên CPU (đã benchmark: ~25
  bài/giây, transformers 5.12.1 + sentencepiece — không cần transformers<5 như docstring vendor
  cũ ghi, đã verify bằng smoke test tải model + encode thật).

## 2. Mục tiêu

Mở rộng `data/external_news_embeddings/raw_cache/news_emb_articles_{source}.parquet` để:
1. Có cache cho 12 nguồn mới (tạo file mới).
2. Bổ sung (KHÔNG ghi đè) các bài mới vào cache của nguồn đã tồn tại.

**KHÔNG** rebuild dual-group panel (`data/features/dual_group_news_panel.parquet`) hay retrain
baseline nào — đó là việc của baseline `2026-07-25_dual_group_news_embedding_baseline` (cache-only
theo thiết kế của baseline đó), chạy riêng sau nếu user muốn.

## 3. Input / Output

- **Input:** `crawl_data/data/*.csv` (qua `discover_source_files()`, read-only import từ
  `2026-07-25_dual_group_news_embedding_baseline/code/vendor_data_eda/discover_news.py`).
- **Output:** parquet files mới/cập-nhật trong `data/external_news_embeddings/raw_cache/`,
  schema giữ nguyên (`url`, `raw_0..raw_767`, + các cột metadata gốc của mỗi source) để tương
  thích 100% với `news_embeddings.py::_get_article_embeddings` (cache-only reader) của baseline
  kia — không đổi format.

## 4. Cô lập (hard isolation, CLAUDE.md §3.F rule 3)

- KHÔNG sửa bất kỳ file nào trong `2026-07-25_dual_group_news_embedding_baseline/`. Import
  read-only (`discover_news.discover_source_files/load_source`, `phobert_embeddings.
  extract_phobert_embeddings`, `news_embeddings.TICKER_PATTERN/_article_cache_path`) — đúng tiền
  lệ đã có trong repo này (`2026-07-18_gated_crossattn_baseline` cũng import read-only từ
  `2026-07-07_embedding_baseline`).
- Danh sách nguồn mới (khach_quan) định nghĩa TRONG baseline này, không sửa
  `KHACH_QUAN_SOURCES`/`TONG_HOP_SOURCES` gốc.

## 5. Success criteria / Go-No-go

- [ ] 12 nguồn mới có file cache parquet, schema đúng (cột `url` + 768 cột `raw_*`).
- [ ] Nguồn đã tồn tại: bài cũ trong cache KHÔNG bị mất/đổi (row count cũ là tập con của row
      count mới).
- [ ] Re-chạy script scoping (đếm NEW-to-encode) sau khi build → 0 cho mọi nguồn.
- [ ] `raw_cache/` được backup trước khi ghi (do đây là dữ liệu tính toán tốn công, khó tái tạo).
- [ ] Test chạy trên 1 nguồn nhỏ (dry-run/smoke) TRƯỚC khi chạy full 13,818 bài.
- [ ] pytest pass cho phần logic upsert (không cần PhoBERT thật trong test — mock/monkeypatch).
- [ ] Code review (adversarial) chạy trước khi coi "done".

## 6. Out of scope

- Rebuild `dual_group_news_panel.parquet` hoặc retrain model nào — follow-up riêng nếu được yêu cầu.
- Phân loại lại toàn bộ nguồn cũ — chỉ thêm 12 nguồn mới vào `khach_quan`.
