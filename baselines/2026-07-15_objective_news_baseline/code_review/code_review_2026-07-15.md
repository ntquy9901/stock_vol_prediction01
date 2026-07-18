# Code Review — Objective News Baseline (2026-07-15)

**Tool:** `/code-review` (effort medium) — 1 finder agent (correctness angles) + self-verify
against real crawl data, per CLAUDE.md Definition of Done ("Code review LUÔN, mọi change").

## Findings (5) — tất cả đã xử lý

| # | Finding | Verdict | Xử lý |
|---|---|---|---|
| 1 | `NAME_ALIASES` regex thiếu `re.IGNORECASE` — bỏ sót brand name viết thường (4× "vinamilk" trong corpus thật) | CONFIRMED | Fixed: thêm `re.IGNORECASE` cho alias pattern |
| 2 | Ticker-code regex ngược lại KHÔNG nên case-insensitive (tự phát hiện khi fix #1) — nguy cơ khớp nhầm từ thường (vd "GAS"~"gas") | CONFIRMED (self) | Fixed: bỏ `re.IGNORECASE` khỏi ticker regex |
| 3 | Tuổi Trẻ (59/59 dòng) + vietstock (174/669 dòng) publish_time rỗng bị drop, log giống hệt "0 ticker match" (không phân biệt được) | CONFIRMED | Fixed: thêm counter `no_date_dropped` riêng trong log. **Đã thử fallback `crawl_time` (rồi REVERT)** — xem mục "Sự cố tự phát hiện" bên dưới |
| 4 | `design.md` ghi có dedup document_id/checksum nhưng code chưa làm | CONFIRMED | Fixed: thêm `_dup_or_id()` — verify: document_id vốn unique 100% trong corpus hiện tại nên không đổi số liệu, nhưng đúng thiết kế và an toàn khi nguồn sau này trùng |
| 5 | Test chưa cover riêng: PCA-fit path, leakage-guard trên file unenriched | PLAUSIBLE | Ghi nhận follow-up (không block done — 7 test hiện tại cover đúng behavior mới sửa) |

## Sự cố tự phát hiện khi fix #3 (nghiêm trọng hơn finding gốc)

Thử fallback `publish_time → crawl_time` (an toàn về leakage) nhưng **REVERT** sau khi phát
hiện: dồn **179/298 record test-period vào đúng 1 ngày** (2026-07-12, ngày crawl) → cơn bão
tin giả 1 ngày, tệ hơn nhiều so với chỉ mất 174 dòng không có ngày. Đây là lỗi do chính fix
gây ra, không phải từ code review agent — phát hiện qua kiểm tra phân phối ngày sau khi áp
dụng fix, TRƯỚC khi train (tránh lãng phí 1 lần train với data sai). Bài học: mọi "phục hồi
data" phải kiểm tra phân phối kết quả trước khi tin, không chỉ tin theo tổng số record tăng.

## Requested-but-out-of-scope

User yêu cầu thêm (giữa review): **incremental extraction** (manifest + persisted PCA +
merge cache) — đã implement riêng ở `extract_objective_embeddings.py` (xem design.md §3b),
không phải finding từ code review nhưng test cùng lượt (2 test mới: skip-already-processed,
merge-new-without-refit).

## Final state

7/7 pytest pass (`pytest baselines/2026-07-15_objective_news_baseline/test/ -v`).
Dry-run thật trên `crawl_data/data/objective/`: 341 record, test-period (2021-2026) 119
record / 113 cặp (stock, ngày), max 7 record/ngày (không còn date-clustering giả).
