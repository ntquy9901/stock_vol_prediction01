# Deliverables 2026-08-15 — Dự đoán độ biến động chứng khoán (VN30 case study)

Gói tài liệu để báo cáo. Kết quả: mô hình đa nhánh (LSTM + GAT có hướng vol→PK + news có gate) được
đánh giá bằng **ablation leave-one-out** so với **HAR**, đa horizon (1/5/10/22), nhiều seed, kiểm định
**Diebold-Mariano đa-metric** (QLIKE, MSE/RMSE/R², MAE). Kết luận chính: **không cấu hình nào vượt HAR
một cách nhất quán/có ý nghĩa trên các loss chuẩn**; HAR và LSTM-only (price-only) là mạnh nhất.

## papers/ — các bản paper
- **`trackA_gat_paper_draft_3seed.md`** ⭐ **BẢN CHÍNH để báo cáo** (3 seed 42/123/2026, mean±std, DM
  đa-metric, đủ 4 horizon).
- `trackA_gat_paper_draft_3seed_vi.md` — **bản tiếng Việt** của bản chính 3-seed.
- `trackA_gat_paper_draft_5seed.md` / `_5seed_vi.md` — **bản 5 seed** (42,123,2026,7,2024), EN + VI (bản mạnh nhất về thống kê; kết luận khớp 3-seed).
- `trackA_gat_paper_draft.md` — bản 1 seed (seed 42), cùng cấu trúc.
- `trackA_gat_paper_draft_3seed_mz.md` — 3 seed + phân tích Mincer-Zarnowitz (hiệu chỉnh/bias); §6.4 DM đa-metric.
- `soict2026_trackA_gat_v1.tex` / `_vi.tex` — LaTeX SOICT 3-seed (EN/VI).
- `soict2026_trackA_gat_v1_5seed.tex` / `_5seed_vi.tex` — LaTeX SOICT 5-seed (EN/VI).
- Mỗi paper .md có kèm **.pdf** cùng tên. `diagrams/` — 2 hình (SVG cho .md, PDF cho .tex).

## explainers/ — tài liệu giải thích phụ
- `data_organization_example.md` — cách tổ chức train/validation/test (chia theo từng mã, snapshot đồ
  thị 1 ngày chung + mask), có ví dụ ngày thật (2024-06-01, 2019-01-02): **không lệch ngày, không leakage**.
- `gat_vol2pk_explained.md` — giải thích + công thức: cạnh có hướng vol→PK lead-lag và multi-head GAT
  tính thế nào.
- `ARCHITECTURE_DETAILED.md` — kiến trúc chi tiết (3 nhánh, 5 node feature, edge, ablation).

## code/ — mã nguồn chính (vừa đủ)
- `model.py` — mô hình (LSTM + GAT + news + gate + sàn dương).
- `gat.py` — lớp multi-head GAT tự viết.
- `edges.py` — dựng cạnh vol→PK Top-5 (train-only, đóng băng).
- `features.py` — 5 node feature (HAR ×3 + market_pk + volume_zscore).
- `train_resume.py` — vòng huấn luyện + early-stopping + resume.
- `run_ablation.py` — ablation leave-one-out (FULL / minus_graph / minus_gate / minus_news + HAR).
- `run_lstm_only.py` — mốc price-only (LSTM-only).
- `run_trackA.py` — dựng basis (5 feature + news + edge) dùng chung.
- `dm_report.py` — Diebold-Mariano đa-metric (QLIKE/SE/AE), đơn/đa seed ensemble.
- `aggregate_seeds.py` — gộp nhiều seed (mean±std).
- `mz_report.py` — hồi quy Mincer-Zarnowitz.

## Ghi chú số liệu
- Target: Parkinson **variance** một ngày tại `t+h` (không phải trung bình h ngày).
- Metric báo cáo: MSE, RMSE, MAE, R², QLIKE (không dùng Directional Accuracy — target đối kháng
  (anti-persistent) nên không có ý nghĩa).
- Universe: 33 mã VN30 (point-in-time). Chia 70/15/15 theo thời gian, theo từng mã; scaler & cạnh
  ước lượng train-only.
