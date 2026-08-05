# Summary — Calendar-Group Ablation + Per-Point delta_QLIKE EDA

**What changed:** mở rộng `baselines/2026-08-01_calendar_news_gate_baseline/` với (1) khả năng
train chỉ 1 SUBSET calendar feature (`--calendar_groups {tet_only,earnings_only,generic_calendar,all}`)
và (2) 1 script EDA mới (`analyze_news_calendar_correlation.py`) tính delta_QLIKE THEO TỪNG ĐIỂM
test (không chỉ trung bình) rồi gộp theo tháng/Tết/mùa BCTC — trả lời trực tiếp câu hỏi "tin tức
có ảnh hưởng theo mùa không" bằng thống kê, tái dùng 2 checkpoint đã train sẵn từ 25/07 (không cần
train model mới cho phần EDA).

## Files

| File | Trách nhiệm |
|---|---|
| `code/calendar_features.py` (mở rộng) | thêm `CALENDAR_FEATURE_GROUPS` (tet_only/earnings_only/generic_calendar/all) |
| `code/dataset_calendar_news.py` (mở rộng) | thêm tham số `calendar_feature_names` — subset 10 cột calendar theo tên, validate tên không hợp lệ |
| `code/train_calendar_news_gate.py` (mở rộng) | thêm CLI `--calendar_groups`, đặt tên output dir theo group |
| `code/analyze_news_calendar_correlation.py` (mới) | EDA: per-point QLIKE(model B) − QLIKE(model A), join calendar bucket, Welch t-test + Pearson |
| `test/test_calendar_features.py` (+4 test) | validate `CALENDAR_FEATURE_GROUPS` (phủ đủ, không chồng lấp) |
| `test/test_dataset_smoke.py` (+4 test) | subset feature giảm đúng n_feat, giá trị đúng cột, tên sai raise ValueError |
| `test/test_analyze_news_calendar_correlation.py` (mới, 10 test) | qlike_pointwise khớp `qlike_loss` gộp, date-extraction khớp thủ công, groupby/correlation logic |

## Tests

**48/48 pytest pass** toàn baseline (35 cũ + 13 mới). Không cần checkpoint thật để test logic EDA
(dùng dữ liệu synthetic) — chỉ cần checkpoint thật khi CHẠY THẬT script (`python
analyze_news_calendar_correlation.py`), đã chạy thành công (xem kết quả bên dưới).

## Kết quả thật

### A. Ablation 3 nhóm calendar feature (10 epoch, cùng seed/panel với đối chứng 26/07)

| Nhóm | Cột | Test DirAcc | Test R² | Test QLIKE | Test RMSE |
|---|---:|---:|---:|---:|---:|
| Đối chứng (không calendar) | 0 | 68.76% | 0.7159 | 0.5497 | 0.002635 |
| Đủ 10 cột | 10 | 68.13% | 0.7117 | 0.5660 | 0.002654 |
| tet_only | 2 | 68.76% | 0.7124 | 0.5640 | 0.002651 |
| earnings_only | 2 | 68.71% | 0.7131 | 0.5501 | 0.002648 |
| generic_calendar | 6 | 68.90% | 0.7121 | 0.5583 | 0.002652 |

Không nhóm nào vượt đối chứng trên cả 4 metric đồng thời. `earnings_only` gần đối chứng nhất trên
QLIKE/R². `generic_calendar` có DirAcc nhỉnh hơn đối chứng (+0.14pp) — trong biên độ nhiễu
single-seed, KHÔNG đủ để kết luận cải thiện thật.

### B. EDA per-point delta_QLIKE (Model A = HAR-only 10ep, Model B = dual-group all-ON 10ep, cùng
test set 164 cửa sổ × 32 mã = 5,248 điểm)

- **Theo tháng:** 11/12 tháng có delta_QLIKE dương (tin tức làm dự báo tệ hơn); chỉ tháng 5 hơi âm
  (không đáng kể). Không tháng nào cho tín hiệu "tin tức giúp ích" rõ ràng.
- **Tết (±10 ngày):** Welch t=-0.816, p=0.415 — không khác biệt có ý nghĩa.
- **Mùa BCTC (20 ngày sau mỗi quý):** Welch t=0.392, p=0.695 — không khác biệt có ý nghĩa.
- **Tương quan Pearson** tet_proximity vs delta_QLIKE: r=-0.019 (p=0.168); earnings_proximity vs
  delta_QLIKE: r=-0.020 (p=0.156). Cả hai gần 0, không có ý nghĩa thống kê.

**Kết luận:** ablation (A) và EDA (B) — 2 phương pháp độc lập, khác cơ chế hoàn toàn — đồng thuận:
không phát hiện được hiệu ứng "tin tức theo mùa BCTC/Tết" với model và dữ liệu hiện tại.

## Code review

Mở rộng code cũ (không sửa hàm hiện có, chỉ thêm tham số optional `calendar_feature_names` +
1 file mới) — rủi ro thấp, đã tự kiểm tra: (1) `_calendar_indices` map đúng theo
`CALENDAR_FEATURE_NAMES.index()`, có test xác nhận subset khớp đúng cột của vector đầy đủ; (2)
`extract_target_dates` trong EDA script được test khớp thủ công với `target_idx = i + seq_length +
forecast_horizon - 1` — cùng công thức dataset gốc dùng để build `y_all`; (3) sanity check
`np.testing.assert_array_almost_equal(targs_n, targs_b_n)` trong script thật để xác nhận Model A và
Model B chạy trên ĐÚNG CÙNG cửa sổ trước khi tính delta (nếu lệch sẽ crash ngay, không âm thầm cho
kết quả sai).

## Impact analysis

Chỉ thêm code trong baseline folder của chính nó (không sửa sibling `dual_group_news_embedding_baseline`
hay `per_ticker_news_gate_baseline`) — dùng lại checkpoint đã có (`models/har_only_ablation_ref_2026-07-25_110813`,
`models/dual_group_news_2026-07-25_011719`) ở chế độ read-only (`torch.load` + `.eval()`).

## DoD checklist

- [x] Code: thêm tham số optional (backward-compatible, default giữ nguyên hành vi cũ)
- [x] Tests: 48/48 pass (13 test mới, bao gồm cả pure-logic EDA test không cần checkpoint)
- [x] Train 3 ablation group + 1 EDA thật (không train thêm cho EDA — tái dùng checkpoint có sẵn)
- [x] Impact analysis
- [x] Summary report (file này) + cập nhật `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` §7
- [ ] Diff-coverage: Not run (tooling gap đã biết)

## Đề xuất tiếp theo

Không nên đầu tư thêm vào calendar feature theo hướng hiện tại (concat tĩnh hay tách nhóm) — cả
ablation và EDA đều cho tín hiệu gần 0. Nếu vẫn muốn theo đuổi giả thuyết thời vụ, cần: (1) nguồn
dữ liệu tin tức phong phú/chính xác hơn (vd ngày công bố BCTC thật/mã thay vì proxy lịch chung),
hoặc (2) multi-seed để loại trừ khả năng single-seed che khuất tín hiệu yếu, trước khi cân nhắc
kiến trúc phức tạp hơn (Cách 2 — gate theo thời gian).
