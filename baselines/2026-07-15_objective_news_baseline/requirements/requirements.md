# Requirements (Specify) — Objective News Baseline

**Baseline:** `2026-07-15_objective_news_baseline` · **Ngày:** 15/07/2026
**Theo quy trình SDD** (CLAUDE.md §1.5) — pha Specify. Clarify đã chốt qua AskUserQuestion (2026-07-15):
scope = vietstock_records + vsdc_records + 5 file news_unenriched; vai trò = baseline mới độc lập.

## 1. Mục tiêu

Thay "báo cáo phân tích tổng hợp" (broker PDF reports — khuyến nghị MUA/BÁN, ý kiến chủ quan của
CTCK) đang dùng trong nhánh news hiện tại bằng **dữ liệu khách quan** (objective): thông báo sự kiện
doanh nghiệp chính thức (cổ tức, phát hành CP, ĐHCĐ, trái phiếu — từ vietstock/VSD) + tin tức phổ
thông đã gắn ticker. Convert sang vector (PhoBERT embedding, cùng recipe với baseline 2026-07-07) và
so sánh DirAcc/QLIKE/R² với:
- HAR-only baseline: 69.98% DirAcc (70 epoch)
- Embedding baseline (báo cáo phân tích): 68.76% DirAcc (40 epoch)
- Latent noise baseline: 69.33% DirAcc (10 epoch)

## 2. Input data (raw, READ-ONLY)

Nguồn: `D:/bmad-projects/crawl_data/data/objective/` (8 file, đã khảo sát 2026-07-15):

| File | Rows | company_code | Cách dùng |
|---|---|---|---|
| `vietstock_records.csv` | 670 | có sẵn | Dùng trực tiếp company_code (filter VN30) |
| `vsdc_records.csv` | 6 | có sẵn | Dùng trực tiếp company_code (filter VN30) |
| `news_unenriched_vnexpress_records.csv` | 89 | RỖNG | Ticker-regex match trên title+raw_text (tái dùng pattern từ `extract_embeddings.py`) |
| `news_unenriched_tuoitre_records.csv` | 60 | RỖNG | nt |
| `news_unenriched_thanhnien_records.csv` | 101 | RỖNG | nt |
| `news_unenriched_vietnamplus_records.csv` | 104 | RỖNG | nt |
| `news_unenriched_nld_records.csv` | 52 | RỖNG | nt |
| `objective_v2026-07-14.csv` | 435 | có sẵn | **SKIP** — đây là snapshot merge CỦA CHÍNH vietstock+vsdc (430+5=435, verify khớp), dùng sẽ double-count |
| `objective_v2026-07-12.csv` | 0 (empty) | — | SKIP |

Schema chung (unified sẵn từ nguồn crawl, KHÔNG cần viết aggregator mới như `aggregate_news_sources.py`):
`document_id, source, source_tier, url, publish_time, crawl_time, company_code, company_name, title,
raw_text, language, category, event_type, attachment_urls, checksum, raw_path`

**Phát hiện quan trọng:** với vietstock/vsdc, `raw_text` == `title` (không có nội dung dài hơn — chỉ
là câu thông báo sự kiện 1 dòng). Với news_unenriched, `raw_text` chứa HTML thô (`<a><img>...`) — cần
strip HTML trước khi tokenize.

## 3. Rủi ro dữ liệu đã kiểm tra

- **Publish_time sau crawl_time (khả năng leakage):** kiểm tra `vietstock_records.csv` → chỉ
  **5/669 dòng (0.7%)** có `publish_time > crawl_time`. Không phải lỗi hệ thống — xử lý bằng cách
  drop 5 dòng này (hoặc clip về crawl_time), không cần redesign.
- **Độ thưa:** 670+6 sự kiện doanh nghiệp trải ~20 năm (2005-2026) / 30 ticker ⇒ cực thưa cho
  vietstock/vsdc riêng lẻ. Cộng thêm ~400 tin phổ thông sau ticker-match (số match thực tế sẽ ít hơn
  400 vì không phải tin nào cũng chứa ticker VN30). **Go/no-go phụ thuộc coverage đo được sau khi
  chạy extraction** — nếu test-period (2021-2026) gần như không có dòng nào, baseline này lặp lại
  kết luận "NO-GO" như `project-sentiment-price-eda-result` (event quá thưa/sạch để học được).

## 4. Success criteria / Go-No-Go

- **Go:** test-period (2021-2026) có ≥ vài chục stock-days có tin (đo được sau extraction) VÀ chạy
  10-epoch được DirAcc không thấp hơn embedding baseline hiện tại (>68.76%) quá nhiều (trong biên độ
  nhiễu ~1%).
- **No-go:** nếu extraction cho thấy test-period gần như trống (giống EDA 2026-07-11) → dừng, ghi
  nhận kết luận, không train tiếp.

## 5. Training policy

Theo CLAUDE.md Training policy: chạy **10 epoch trước** (không phải 20 — đây là thí nghiệm MỚI,
khác với latent-noise đã được approve riêng cho 20 epoch). Báo cáo kết quả 5/10 epoch trước khi xin
approve train dài hơn.

## 5b. Addendum (2026-07-15, sau khi chạy dry-run thật)

- **Ticker-code regex trên tin phổ thông gần như vô dụng:** chỉ 2/402 dòng `news_unenriched_*`
  khớp bằng mã ticker — tin tức thường viết tên thương hiệu ("Vinamilk", "Vinfast"), không viết mã
  ("VNM"). Theo yêu cầu user: **không bắt buộc tin phải chứa mã cổ phiếu**, chỉ cần convert tin
  thành vector rồi xem nó ảnh hưởng mã nào.
- **Đã thêm `NAME_ALIASES`** (dict ticker → tên thương hiệu, curate thủ công cho 30 mã VN30, vd
  VNM→"Vinamilk", VIC→"Vingroup"/"Vinfast" — Vinfast là công ty con Vingroup nên route về VIC) —
  match theo tên thương hiệu SONG SONG với ticker-regex (union 2 tập kết quả).
- **Kết quả sau khi thêm alias:** 332→340 record, test-period 105→112 stock-day. Uplift nhỏ —
  nguyên nhân chính KHÔNG phải do cách match, mà do **volume crawl quá nhỏ** của 5 nguồn tin phổ
  thông (52-104 dòng/nguồn) — trần dữ liệu, không phải trần thuật toán match.
- **Giới hạn đã biết:** `NAME_ALIASES` là danh sách thủ công, không đầy đủ (không cover mọi biến
  thể viết tắt/tên cũ), match theo substring không phân biệt hoa-thường (case-insensitive, vì brand
  name có thể viết thường trong tin tức — vd "vinamilk" 4 lần trong corpus).

## 5c. Code review (2026-07-15, `/code-review` effort medium, 1 agent + self-verify)

5 finding, tất cả xử lý:
1. **Alias case-sensitivity** (CONFIRMED) — `NAME_ALIASES` regex thiếu `re.IGNORECASE` nên bỏ sót
  4 lần "vinamilk" viết thường trong corpus thật. **Fixed**: thêm `re.IGNORECASE` cho alias pattern.
2. **Ticker-code regex ngược lại KHÔNG nên case-insensitive** (finding liên quan, tự phát hiện khi
  fix #1) — mã ticker luôn viết HOA trong báo tài chính VN; case-insensitive có nguy cơ khớp nhầm từ
  thường (vd "GAS" ~ "gas" trong tin nấu ăn). **Fixed**: bỏ `re.IGNORECASE` khỏi ticker regex (alias
  vẫn case-insensitive vì đó là brand name, không phải từ thường).
3. **Tuổi Trẻ (59/59 dòng) bị drop hoàn toàn do publish_time rỗng, log nhìn giống "0 ticker match"**
  (CONFIRMED, kiểm tra lại thấy vietstock_records.csv cũng có 174/669 dòng publish_time rỗng — vấn đề
  rộng hơn ban đầu tưởng). **Đã thử fallback `publish_time → crawl_time`** (an toàn về leakage vì
  crawl_time luôn ≥ ngày công bố thật) **nhưng REVERT** — kiểm tra thấy fallback dồn **179/298 record
  test-period vào đúng 1 ngày** (ngày crawl 2026-07-12), tạo cơn bão tin giả tệ hơn mất data. **Fix
  cuối cùng**: vẫn drop, nhưng thêm counter `no_date_dropped` in ra log riêng — không còn lẫn với "0
  ticker match" (minh bạch, không cố phục hồi bằng data giả).
4. **Design/code mismatch**: `design.md` nói có dedup document_id/checksum nhưng code chưa có.
   **Fixed**: thêm `_dup()` dedup theo `document_id` (fallback `checksum`) — kiểm tra thực tế:
   document_id vốn đã unique 100% trong corpus hiện tại nên dedup không đổi số liệu, nhưng đúng với
   thiết kế đã ghi và an toàn nếu nguồn sau này có trùng.
5. **Test coverage gap** (PCA path chưa test riêng, unenriched leakage-guard chưa test riêng) —
   ghi nhận là follow-up, không block done (coverage hiện tại: 5 test pass, cover đúng hành vi mới
   sửa — brand-name match, leakage-guard trên direct file).

**Số liệu cuối cùng (sau tất cả fix), dry-run thật trên `crawl_data/data/objective/`:**
vietstock kept=325 (không đổi so với trước khi thêm alias — no_date_dropped=174 minh bạch),
vsdc kept=5, unenriched ticker/name-matched: vnexpress=10, tuoitre=0 (toàn bộ no_date_dropped=59),
thanhnien=0, vietnamplus=0, nld=1. **Tổng 341 record, test-period (2021-2026) = 119 record / 113
cặp (stock, ngày) — không còn date-clustering giả** (max 7 record/ngày, so với 179 record/ngày ở
bản fallback bị revert).

## 6. Out of scope

- Không viết lại `aggregate_news_sources.py` (khác schema, khác mục đích — objective data đã unified
  sẵn từ nguồn).
- Không sửa `src/`, không sửa baseline khác (hard isolation §3.F.3).
- Không dùng `macro/raw/*.csv` (dxy, sbv_policy_rates) — đó là numeric macro series, không phải text
  để "convert sang vector", ngoài phạm vi yêu cầu này.

## 7. Closure decision (2026-08-03)

**Status: đóng (closed), không tiếp tục sang bước train.** Quyết định trong phiên audit toàn diện
trước khi viết paper, dựa trên các điểm sau:

- Extraction đã hoàn tất và đạt go-criterion riêng của baseline này (§4: "≥ vài chục stock-days có
  tin"): 341 record, 113 stock-day trong test-period (2021-2026) — không phải "no-go" theo nghĩa dữ
  liệu quá thưa, khác với các baseline khác đã archive vì lý do "null/rejected result".
- Tuy nhiên: chưa có bước train (không có `train_*.py`/`model_*.py`/`results.json` nào trong
  `code/`) — baseline dừng lại ở bước extraction, chưa từng thực sự so sánh DirAcc/QLIKE với 3
  baseline tham chiếu nêu ở §1.
- Bối cảnh quyết định: dự án đã có ~10 kết quả null từ các news-baseline khác cùng lớp vấn đề (macro
  news, sentiment decay, dual-group embedding, v.v. — xem memory
  `project_null_result_pattern_and_sota_pivot`), và 341 record/113 stock-day là volume rất nhỏ so
  với ~5200 test-window points của pipeline chính — tiên nghiệm rủi ro null cao, giống pattern đã
  thấy lặp lại nhiều lần.
- Deadline nộp paper trong tháng tới; ưu tiên hiện tại đã chuyển sang: (a) đảm bảo tính đúng đắn của
  các bug đã audit trên pipeline chính (P1.1/P1.2/P1.3/DirAcc), và (b) train lại đúng 1 lần cuối
  cùng pipeline headline (per-ticker news-gate) 20 epoch so với HAR-only để có bảng kết quả cuối
  cùng cho paper — không còn ngân sách thời gian để mở thêm 1 baseline thử nghiệm mới trước hạn nộp.

**Không archive folder này** (khác với các baseline "null/rejected" đã archive) — giữ nguyên tại chỗ
vì đây là "chưa hoàn thành do ưu tiên thời gian", không phải "đã kết luận và bị bác bỏ". Extraction
code (`extract_objective_embeddings.py`) và test đã pass vẫn có giá trị tham khảo nếu muốn tiếp tục
sau khi nộp paper. Không có claim nào từ baseline này được đưa vào paper.
