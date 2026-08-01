# Code Review — Calendar-Augmented News Gate Baseline (2026-08-01)

**Phương pháp:** tự thực hiện review đối kháng (agent-based/interactive PR không khả thi cho
phiên làm việc này, cùng lý do đã ghi ở các baseline trước — xem
`docs/report_2026-07-25/BAO_CAO_CHO_THAY.md` §4.1) — 3 lớp theo CLAUDE.md §5: Blind Hunter (giả
định chắc chắn có bug), Edge Case Hunter (đi qua từng nhánh điều kiện), Acceptance Auditor (đối
chiếu requirements.md).

**Phạm vi:** 3 file mới — `calendar_features.py`, `dataset_calendar_news.py`,
`train_calendar_news_gate.py`. Model (`PerTickerGatedNewsBaseline`) và loader dual-group
(`load_news_panel`) KHÔNG sửa (import read-only) — đã review ở baseline trước, không review lại.

## Kiểm tra đặc biệt: rủi ro rò rỉ dữ liệu (data leakage)

Bài học từ `2026-07-25_dual_group_news_embedding_baseline` (bug `TRAIN_CUTOFF` PCA fit lẫn
val/test): mọi bước "fit" theo train-set đều phải soi kỹ. **`compute_calendar_vector` KHÔNG có
bước fit nào** — là hàm thuần túy của ngày, không phụ thuộc train/val/test split, không có tham số
học từ dữ liệu. Do đó **không có đường nào để rò rỉ dữ liệu tương lai vào feature này** — khác hẳn
lớp bug đã gặp trước đây. Đã xác nhận bằng cách đọc lại toàn bộ `calendar_features.py`: không có
biến global nào được "fit" hay cache theo lần gọi trước ngoài `_calendar_cache` (dataset-level,
chỉ cache KẾT QUẢ của hàm thuần, không đổi giá trị trả về).

## Findings

### MEDIUM — "Cuối tháng/quý" dùng ngày lịch, không phải ngày giao dịch thật (đã biết trước, ghi trong requirements A3)
`_is_month_end` dùng `calendar.monthrange` (ngày dương lịch), không tra lịch giao dịch thực tế của
HOSE/HNX. Nếu 3 ngày cuối tháng dương lịch rơi vào cuối tuần/nghỉ lễ, ngày giao dịch cuối cùng thật
sẽ lệch vài ngày so với flag này. **Quyết định giữ nguyên** (Simplicity Gate, đã nêu rõ trong
requirements.md A3) — không fix trong baseline này, ghi nhận là giới hạn đã biết.

### LOW — Bảng `TET_DATES` chỉ phủ 2005-2027
Nếu tái sử dụng module này cho dữ liệu ngoài khoảng này, hàm vẫn chạy (không crash, đã test —
`test_year_without_explicit_tet_entry_still_works`) nhưng `tet_proximity`/`in_tet_window` sẽ sai
(khoảng cách tính tới năm gần nhất trong bảng, không phải Tết thật của năm đó). Không ảnh hưởng
baseline này (dữ liệu giá 2006-11-21 → 2026-06-09, nằm gọn trong bảng). Flag cho người dùng lại
module này trong tương lai.

### LOW — Ranh giới `in_earnings_window`/quarter-end không đối xứng
Cửa sổ định nghĩa `(quarter_end, quarter_end + 20]` — ngày quarter-end chính nó KHÔNG được tính là
"trong mùa BCTC" (vd 2020-06-30 → `in_earnings_window=0`, `is_quarter_end=1` cùng lúc). Đây là quy
ước có chủ đích (ngày kết thúc quý là ngày CHỐT SỔ, chưa phải ngày công bố), không phải bug, nhưng
nêu rõ để tránh hiểu lầm khi đọc `results.json`/feature values sau này.

### Không tìm thấy HIGH nào
Đã kiểm tra kỹ: (1) thứ tự cột `CALENDAR_FEATURE_NAMES` khớp đúng thứ tự trả về trong
`compute_calendar_vector` (test `test_calendar_values_match_pure_function` xác nhận qua dataset
thật); (2) `_calendar_cache` không rò rỉ giữa train/val/test (mỗi Dataset instance có cache riêng,
không phải biến module-level); (3) dtype nhất quán float32 xuyên suốt (`dual_vec` + `cal_vec`
concat không lỗi dtype); (4) không có bare `except`, không mutable default arg trong 3 file mới.

## Acceptance Audit (đối chiếu requirements.md §5)

| Tiêu chí | Trạng thái |
|---|---|
| `compute_calendar_vector` đúng công thức, unit test pass | ✅ 23/23 test pass |
| Dataset x_news shape đúng, không NaN/Inf, coverage log | ✅ (synthetic + real-data-slice test) |
| Model forward+backward không NaN với n_feat rộng hơn | ✅ `test_train_epoch_smoke_end_to_end` |
| Train 10 epoch thật, so sánh với per_ticker_gate (không calendar) | ⏳ chưa chạy tại thời điểm review này — chạy NGAY SAU review (thứ tự CLAUDE.md: review trước train thật, nhưng review code KHÔNG phụ thuộc kết quả train để đánh giá đúng/sai code) |
| pytest toàn bộ pass | ✅ 35/35 |

## Kết luận

Không có HIGH/MEDIUM nghiêm trọng cần fix trước khi train. 1 MEDIUM (calendar-day proxy cho
month/quarter-end) và 2 LOW đã ghi nhận là giới hạn CÓ CHỦ ĐÍCH (đã nêu trong requirements.md),
không phải lỗi cần sửa ngay — phù hợp Simplicity Gate cho baseline thử nghiệm 10-epoch này.
