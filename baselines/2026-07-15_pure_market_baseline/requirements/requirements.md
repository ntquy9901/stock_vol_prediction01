# Requirements (Specify) — Pure Market-Vector Baseline

**Baseline:** `2026-07-15_pure_market_baseline` · **Ngày:** 15/07/2026
**Theo SDD** (CLAUDE.md §1.5). User request (nguyên văn ý tưởng): "nếu mỗi ngày, tin tức lấy hết
xem như là các vectors của thị trường, không phân biệt vectors nào của cổ phiếu nào, đưa các
vectors thị trường vào model theo từng ngày."

## 1. Ý tưởng / Giả thuyết

Tất cả 2 baseline news trước (`2026-07-07_embedding_baseline`, `2026-07-15_objective_news_baseline`)
đều yêu cầu match bài viết → 1 mã cổ phiếu cụ thể (ticker/company_code/brand-alias). Root cause đã
xác định ở deep-dive (`docs/reports/2026-07-15_deep_dive_objective_news_baseline.md`): việc match
theo mã làm coverage cực thấp (0.2-5.5%) → nhánh news suy biến gần hằng số.

**Ý tưởng mới:** BỎ HẲN yêu cầu match theo mã. Mỗi ngày, gom TẤT CẢ bài viết (bất kể nói về mã
nào, kể cả tin vĩ mô/chung chung) thành **1 vector thị trường** duy nhất cho ngày đó, rồi
**broadcast (phát) vector này cho MỌI cổ phiếu** trong ngày đó — không gate, không phân biệt mã
nào "được" nhận vector, mã nào không. Test giả thuyết: tín hiệu "không khí thị trường chung" ngày
hôm đó có giúp dự báo volatility từng mã hay không, dù không biết bài viết nói về mã nào.

**Khác với `2026-07-08_market_fallback`:** baseline đó GATE (chỉ dùng market vector khi mã KHÔNG
có tin riêng, còn có tin riêng thì dùng tin riêng) — vẫn giữ nhánh per-stock ticker-matched. Baseline
NÀY bỏ hẳn nhánh per-stock, market vector là NGUỒN NEWS DUY NHẤT, dùng cho MỌI mã MỌI ngày.

## 2. Input data — TÁI DÙNG, KHÔNG extract lại

`data/sentiment_embedding/market_emb.npz` — đã tồn tại sẵn từ baseline `2026-07-08_market_fallback`
(`extract_market_embeddings.py`, PhoBERT trên TOÀN BỘ `unified_articles.csv`, không lọc ticker,
PCA 768→64 fit train-only). Đọc read-only, KHÔNG sửa/extract lại.

**Coverage đo được (dry-run):** 3668 ngày có tin trên toàn lịch sử, **1379 ngày** trong giai đoạn
2021+ — dày hơn nhiều so với 2 hướng match-theo-mã trước (0.2-5.5%).

## 3. Success criteria / Go-No-Go

- **Go:** test DirAcc ≥ HAR-only (69.98%) HOẶC ít nhất vượt rõ ràng 2 biến thể match-theo-mã
  trước (68.76% / 67.87%) ở cùng ngân sách epoch (10).
- **No-go:** nếu thấp hơn cả 2 biến thể trước → xác nhận việc BỎ ticker-matching không tự động
  giải quyết vấn đề (vd nhiễu từ tin không liên quan lấn át tín hiệu).

## 4. Training policy

10 epoch trước (thí nghiệm mới, độc lập, theo CLAUDE.md Training policy) — báo cáo trước khi xin
train dài hơn.

## 5. Out of scope

- Không sửa `extract_market_embeddings.py` hay bất kỳ file nào của `2026-07-08_market_fallback`
  (hard isolation §3.F.3) — chỉ đọc output `.npz` của nó.
- Không thêm gate/per-stock news branch (đó là baseline khác, đã có).
