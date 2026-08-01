# Requirements — Calendar-Augmented News Gate Baseline

## 1. Mục tiêu

User giả thuyết: ảnh hưởng của tin tức lên biến động (volatility) có thể thay đổi theo thời điểm
trong năm — mạnh hơn quanh mùa công bố báo cáo tài chính (BCTC) quý, và quanh Tết Nguyên Đán (nghỉ
dài, thanh khoản/tâm lý thị trường thay đổi). Hiện kiến trúc KHÔNG có bất kỳ feature thời gian dạng
lịch nào (xem báo cáo `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §4) — chỉ có HAR rolling-window
và thứ tự bước LSTM.

**Approach đã chọn (Cách 1, user quyết định 2026-08-01):** nối thêm calendar feature vào `x_news`
như feature phụ, KHÔNG đổi kiến trúc model — gate vẫn tĩnh theo mã như baseline hiện tại
(`per_ticker_news_gate_baseline`). Đây là bước rẻ để xem có tín hiệu gì trước khi cân nhắc leo
thang lên gate-theo-thời-gian (Cách 2, KHÔNG làm trong baseline này).

## 2. Base architecture (user quyết định, xem AskUserQuestion 2026-08-01)

Kế thừa từ `2026-07-26_per_ticker_news_gate_baseline` (KHÔNG phải bản không-gate) — dùng thẳng
`PerTickerGatedNewsBaseline` **KHÔNG SỬA** (n_feat là tham số constructor, tự nhận input rộng hơn).

**Đánh đổi đã biết (nêu rõ theo yêu cầu user khi chọn option này):** so sánh baseline này với
`per_ticker_news_gate_baseline` (không calendar) sẽ trộn 2 biến (gate + calendar) nếu so với
`dual_group_news_embedding_baseline` gốc — nhưng vì CẢ HAI đều dùng chung `PerTickerGatedNewsBaseline`
với gate giữ nguyên cơ chế, phép so sánh ĐÚNG (cô lập được biến "có/không calendar feature") là so
với chính `per_ticker_news_gate_baseline` (cùng epoch, cùng mọi thứ, chỉ khác `x_news` rộng hơn).

## 3. Input/Output

**Input:** `x_har [B,22,30,3]`, `adj [B,30,30]`, `x_news [B,22,30,146+10=156]` (146 cột dual-group
cũ + 10 cột calendar mới), `y [B,30]` — hình dạng giống hệt sibling, chỉ khác chiều cuối `x_news`.

**Calendar feature (10 cột), tính THUẦN từ ngày (date string), KHÔNG cần dữ liệu ngoài, broadcast
giống nhau cho mọi mã trong cùng 1 ngày** (giống cách `macro_news_baseline` broadcast macro vector
— xem `code/vendor` pattern đã có):

| # | Tên cột | Công thức | Ý nghĩa |
|---|---|---|---|
| 1 | `dow_sin` | sin(2π·weekday/5) | thứ trong tuần (Mon=0..Fri=4), chu kỳ 5 ngày giao dịch |
| 2 | `dow_cos` | cos(2π·weekday/5) | (cùng trên, phần cos) |
| 3 | `month_sin` | sin(2π·(month-1)/12) | tháng trong năm, cyclical |
| 4 | `month_cos` | cos(2π·(month-1)/12) | (cùng trên, phần cos) |
| 5 | `tet_proximity` | exp(-\|days_to_nearest_tet\|/10) | tín hiệu liên tục, =1 đúng ngày Tết, giảm dần ±10 ngày |
| 6 | `in_tet_window` | 1 nếu \|days_to_nearest_tet\|≤10 else 0 | cờ nhị phân, cửa sổ ±10 ngày quanh Tết |
| 7 | `is_month_end` | 1 nếu ngày ∈ 3 ngày cuối tháng (lịch) else 0 | proxy cuối tháng (KHÔNG dùng lịch giao dịch thật, chỉ ngày lịch — đơn giản hoá có chủ đích) |
| 8 | `is_quarter_end` | `is_month_end` AND month ∈ {3,6,9,12} | proxy cuối quý |
| 9 | `earnings_proximity` | exp(-\|days_to_nearest_deadline\|/10) | deadline = {20/1, 20/4, 20/7, 20/10} mỗi năm (quy định công bố BCTC quý trong 20 ngày) |
| 10 | `in_earnings_window` | 1 nếu ngày ∈ [quý_end, quý_end+20] (bất kỳ quý nào) else 0 | cờ "đang trong mùa công bố BCTC" |

## 4. Giả định (assumptions) — nêu rõ để user/thầy duyệt, KHÔNG tự ý coi là sự thật tuyệt đối

- **A1 — Bảng ngày Tết:** dùng bảng tra cứu ngày Tết Nguyên Đán (dương lịch) 2005–2027, lấy từ
  kiến thức công khai đã biết (không tính toán âm lịch bằng thư viện) — cần user spot-check vài
  mốc quan trọng (đặc biệt các năm dữ liệu dày: 2020–2026) trước khi tin tưởng kết quả.
- **A2 — Earnings-window là PROXY LỊCH CHUNG cho toàn thị trường**, KHÔNG phải ngày công bố BCTC
  thật của từng mã (project chưa có nguồn dữ liệu đó). Deadline 20 ngày sau quý dựa theo quy định
  công bố BCTC quý (chưa kiểm chứng lại với Thông tư/quy định UBCKNN mới nhất — nếu thầy có nguồn
  chính xác hơn, cần cập nhật).
- **A3 — "Cuối tháng" dùng ngày LỊCH** (3 ngày cuối tháng dương lịch), không phải ngày giao dịch
  cuối cùng thật (đơn giản hoá — Simplicity Gate, tránh phải tra lịch giao dịch riêng).
- **A4 — KHÔNG có feature riêng theo per-ticker earnings date** (mỗi mã công bố BCTC ngày khác
  nhau thật) — ngoài scope baseline này (per §Out of scope).

## 5. Acceptance criteria (go/no-go)

- [ ] `compute_calendar_vector(date_str)` trả đúng 10 giá trị, đúng công thức, unit test pass cho
      các mốc đã biết trước (vd 2024-02-10 = đúng ngày Tết → `tet_proximity`≈1.0, `in_tet_window`=1).
- [ ] Dataset trả `x_news` shape `[22,30,156]`, không NaN/Inf, coverage log rõ ràng (giống sibling).
- [ ] Model (tái dùng `PerTickerGatedNewsBaseline` không sửa) forward+backward không NaN với
      `n_feat=156`.
- [ ] Train 10 epoch thật (cap theo Training policy), so sánh DirAcc/R²/QLIKE/RMSE với
      `per_ticker_news_gate_baseline` KHÔNG calendar (CÙNG 10 epoch, CÙNG panel dual-group, CÙNG
      loss MSE) — cô lập đúng 1 biến: có/không 10 cột calendar.
- [ ] Toàn bộ pytest (`test/`) pass.
- [ ] Code review adversarial (tự thực hiện, không có PR tương tác) — HIGH/MEDIUM đã fix.

**Go/no-go:** đây là baseline THỬ NGHIỆM (per Training policy, 10 epoch, chưa cần vượt baseline
nào để coi là "hoàn thành") — mục tiêu là ĐO xem thêm calendar feature có thay đổi metric hay
không, KHÔNG bắt buộc phải thắng để "done". Kết quả no-lift vẫn là kết quả hợp lệ, ghi trung thực.

## 6. Out of scope (KHÔNG làm trong baseline này)

- Cách 2 (gate học theo thời gian, thay `gate_logits` tĩnh bằng hàm của calendar feature) — chờ
  kết quả Cách 1 trước khi quyết định leo thang.
- Ngày công bố BCTC thật theo từng mã (per-ticker real earnings date).
- Train >10 epoch (cần user approve dựa trên kết quả 10-epoch, theo CLAUDE.md Training policy).
- Sửa `dual_group_news_panel.parquet` hay bất kỳ file nào của 2 sibling baseline.
