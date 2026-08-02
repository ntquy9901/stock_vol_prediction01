# Requirements (Specify) — Top-3 News Gate Baseline

**Baseline:** `2026-07-25_top3_news_gate_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** `docs/suggestion/2026-07-25_professor_report.md` §4 "Nhóm 1: Hưởng lợi nhiều nhất"
(Avg ΔR² qua 4 horizon, HGB/XGBoost) + quyết định user 2026-07-25.

## 1. Vấn đề đang giải quyết

Baseline liền trước (`2026-07-25_selective_news_gate_baseline`, 22 mã ON theo ΔR²@t+5) cho kết
quả **bác bỏ giả thuyết**: nhóm ON (46.29% DirAcc) thấp hơn nhóm OFF (51.60%) — khả năng do danh
sách 22 mã quá rộng, lẫn cả tín hiệu yếu/nhiễu từ EDA (nhiều mã chỉ ΔR² ~0.01-0.1, gần ranh giới
nhiễu).

**Giả thuyết mới, thu hẹp hơn:** chỉ bật news cho **3 mã có tín hiệu MẠNH NHẤT và ổn định nhất**
(loại SHB — đã biết nghi vấn spurious/time-proxy) trong bảng "Nhóm 1" của EDA (Avg ΔR² > 0.5,
qua CẢ 4 horizon, không chỉ t+5) — thu hẹp tối đa để giảm nhiễu từ các mã tín hiệu yếu.

## 2. Danh sách mã (user chỉ định trực tiếp)

**NEWS-ON (3 mã):**
| Ticker | Avg ΔR² (4 horizon) |
|---|---|
| VIB | +0.914 |
| ACB | +0.707 |
| MWG | +0.560 |

**NEWS-OFF / bias=0 (29 mã còn lại)** trong 32 mã pipeline thực tế train (xem
`2026-07-25_selective_news_gate_baseline`'s discovery: pipeline dùng 32 mã, không phải 30 như
EDA) — TẤT CẢ mã khác VIB/ACB/MWG, bao gồm cả SHB (dù Avg ΔR² Nhóm 1 cao nhất +3.124, user đã
loại vì nghi time-proxy ở lần trước) và VPB/VRE (không có trong EDA gốc).

## 3. Kiến trúc — tái dùng 100% pattern từ baseline liền trước

Subclass `SelectiveGateNewsBaseline` (từ `2026-07-25_selective_news_gate_baseline`, đọc
read-only) — CHỈ đổi 2 hằng số `NEWS_ON_TICKERS`/`NEWS_OFF_TICKERS`. Không đổi cơ chế mask
(vẫn nhân sau LSTM, trước concat, vẫn buffer không học).

## 4. Success criteria (go/no-go)

- **Go:** train ổn định 10 epoch, mask đúng (test kế thừa), đủ 6 metric.
- **So sánh:** DirAcc nhóm ON (3 mã) vs nhóm OFF (29 mã) vs baseline liền trước (22-mã ON, đã bác
  bỏ) vs baseline không mask (68.50-68.71%) vs HAR-only (69.98%).
- **Diễn giải:** nếu nhóm ON (3 mã) vẫn KHÔNG vượt nhóm OFF → củng cố thêm bằng chứng rằng tín
  hiệu ΔR² từ HGB/XGBoost không chuyển giao được sang kiến trúc LSTM-GNN chung, bất kể thu hẹp
  danh sách bao nhiêu. Nếu VƯỢT → gợi ý ngưỡng ΔR² cần rất cao (>0.5, không phải >0.01) mới có ý
  nghĩa khi chuyển sang kiến trúc khác.

## 5. Training policy

10 epoch thử nghiệm đầu (CLAUDE.md). Không tự train thêm nếu chưa có xác nhận.
