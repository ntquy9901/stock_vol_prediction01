# Báo cáo So sánh Toàn bộ Baseline — Dự báo Biến động VN30

**Ngày:** 25/07/2026
**Mục đích:** Tổng hợp TẤT CẢ baseline đã thử nghiệm trong project, so sánh trên đầy đủ 6 chỉ số
bắt buộc (MSE, RMSE, MAE, R², QLIKE, Directional Accuracy), phục vụ báo cáo cho giảng viên hướng
dẫn.

**Cách đọc báo cáo này:** Project được chia làm 2 nhánh thử nghiệm chính, không hoàn toàn cùng
một trục so sánh (khác nhau về split ratio/epoch/dữ liệu ở một số thời điểm — đã ghi rõ caveat ở
mỗi bảng):

- **Phần A** — Tiến hoá kiến trúc LÕI (không dùng tin tức): Linear → LSTM → LSTM+GNN.
- **Phần B** — Các biến thể TÍCH HỢP TIN TỨC vào nhánh HAR/LSTM-GNN tốt nhất ở Phần A.

---

## Phần A — So sánh kiến trúc lõi (không dùng tin tức)

**Nguồn:** `docs/report_2026-06-27/01_main_report/MODEL_COMPARISON_FINAL_REPORT.md` (2026-06-21),
dataset 30 mã VN30, 99,794 mẫu, temporal split 70/15/15 (không leakage).

| Xếp hạng | Model | MSE | RMSE | MAE | R² | QLIKE | Dir Acc | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **Parallel LSTM-GNN (k-NN)** | 7.024e-06 | 0.002650 | 0.000736 | **0.711** 🏆 | 0.779 | **69.61%** 🏆 | 39 phút train, early-stop epoch 33/48 |
| 🥈 2 | Enhanced LSTM-HAR (anti-overfit) | 3.107e-07 | 0.000557 | 0.000259 | 0.098 | **0.641** 🏆 | 48.56% | Val-test gap <0.094 mọi metric — ổn định nhất |
| 🥉 3 | LSTM-HAR (VN30) | 3.120e-07 | 0.000559 | 0.000297 | 0.161 | 0.566 | 67.39% | ⚠️ Nghi ngờ leakage (hội tụ nhanh bất thường, chỉ 16 epoch) |
| 4 | HAR-R Linear | **2.631e-07** 🏆 | **0.000513** 🏆 | **0.000257** 🏆 | 0.105 | 1.298 | 51.53% | Train tức thời (0.004s), baseline cổ điển |
| 5 | Simple LSTM (Val split) | 0.000105 | 0.010257 | 0.004641 | -0.116 | 2534.6 | 48.50% | ❌ Overfitting nặng — KHÔNG dùng |

**Nhận xét:**
- Không có model nào thắng ở TẤT CẢ metric — có sự đánh đổi (trade-off) rõ giữa nhóm model
  tuyến tính/nông (MSE/RMSE/MAE thấp nhất, nhưng R²/DirAcc kém) và nhóm sâu (Parallel LSTM-GNN:
  R²/DirAcc vượt trội nhưng RMSE/MSE cao hơn — do học trên scale đã chuẩn hoá, biến thiên dự báo
  đa dạng hơn thay vì gần-hằng-số).
- **Parallel LSTM-GNN (k-NN)** là model đầu tiên vượt mốc mục tiêu DirAcc >55%, được chọn làm
  kiến trúc HAR/nhánh nền cho TẤT CẢ baseline tích hợp tin tức ở Phần B.
- **LSTM-HAR (VN30)** có dấu hiệu leakage, không nên dùng làm baseline tham chiếu chính thức.

---

## Phần B — Các biến thể tích hợp tin tức (nhánh HAR/LSTM-GNN + news)

**Nguồn:** `docs/reports/2026-07-18_master_report_sota_news_fusion_baselines.md` (9 baseline đầu,
2026-07-18) + kết quả baseline mới nhất `2026-07-25_dual_group_news_embedding_baseline` (báo cáo
riêng: `docs/reports/2026-07-25_0131_summaryOfUpdate_report.md`).

**Caveat quan trọng (đã ghi trong báo cáo gốc):** các baseline này KHÔNG cùng epoch, KHÔNG hoàn
toàn cùng snapshot dữ liệu tin tức (data được refresh/backfill nhiều lần trong quá trình project
chạy — 2026-07-07 → 07-15 → 07-18 → 07-24/25), và dùng 2 nguồn embedding khác nhau (xem cột
"Nguồn embedding"). So sánh mang tính tương đối, không phải kiểm định thống kê chặt.

| Baseline | Nguồn embedding | Epoch | Test DirAcc | Test R² | Test QLIKE | Test RMSE |
|---|---|---|---|---|---|---|
| **HAR-only** (không tin, tham chiếu) | — | 70 | **69.98%** 🥇 | — | — | — |
| Latent noise | PCA-64 (~3.4K bài) | 10 | 69.33% 🥈 | 0.713 | 0.544 | — |
| Gated Cross-Attn (đã fix bug) | PCA-64 (4.4K bài, refresh) | 15 | 68.97% 🥉 | **0.716** 🏆 | 0.557 | 0.002636 |
| Pure market (broadcast) | PCA-64, market-wide | 10 | 68.95% | 0.713 | 0.556 | — |
| Alignment Loss | PCA-64 (4.4K bài, refresh) | 15 | 68.76% | 0.711 | 0.546 | 0.002656 |
| Embedding baseline (gốc) | PCA-64 (~3.4K bài) | 40 | 68.76% | — | 0.553 | — |
| **Dual-group + EWMA** ⭐ MỚI 07-25 | PCA-32×2 nhóm + EWMA (đã fix leakage) | **10** | **68.50%** | 0.716 | 0.565 | 0.002636 |
| **Dual-group + EWMA** ⭐ MỚI 07-25 | (như trên) | 20 | 68.25% | 0.714 | 0.556 | 0.002642 |
| **Dual-group + EWMA** ⭐ MỚI 07-25 | (như trên) | 40 (early-stop ep36) | **68.71%** | 0.715 | 0.546 | 0.002640 |
| Market fallback (gate cứng) | PCA-64 (cũ) | 37 | 68.69% | 0.706 | 0.548 | — |
| REST-TS | PCA-64 (4.4K bài, refresh) | 15 | 68.29% | 0.706 | **0.543** 🏆 | 0.002680 |
| Objective news | Sự kiện DN + brand-match | 10 | 67.87% | 0.714 | 0.565 | — |

**Đọc bảng theo từng metric:**
- **DirAcc:** HAR-only vẫn đứng đầu tuyệt đối (69.98%) — **CHƯA có biến thể tích hợp tin tức nào
  vượt qua**, kể cả baseline dual-group+EWMA đã train đủ 40 epoch (68.71%, vẫn đứng giữa bảng,
  xấp xỉ Market Fallback 68.69%). Latent-noise là biến thể tin tức gần HAR-only nhất (69.33%,
  cách 0.65 điểm %).
- **R² cao nhất:** Gated Cross-Attn (0.716) và Dual-group+EWMA (0.714-0.716 tuỳ epoch) đồng hạng
  nhất trong nhóm có tích hợp tin tức.
- **QLIKE thấp nhất (tốt nhất, chuẩn học thuật volatility):** REST-TS (0.543) vẫn giữ kỷ lục,
  nhưng Dual-group+EWMA ở 40 epoch đạt 0.546 — sát nút, tốt hơn hẳn bản 10/20 epoch (0.556-0.565)
  và tốt hơn Embedding-baseline gốc (0.553).
- **10 → 20 → 40 epoch (Dual-group+EWMA, đã train theo yêu cầu user để kiểm tra hội tụ):** DirAcc
  dao động trong biên độ hẹp không có xu hướng rõ (68.50%→68.25%→68.71%), 40-epoch dừng sớm
  (early-stop) ở epoch 36 vì val_loss không cải thiện thêm 15 epoch liên tiếp — xác nhận model đã
  hội tụ, KHÔNG còn dư địa cải thiện DirAcc bằng cách train dài hơn với kiến trúc hiện tại. Điểm
  cải thiện thật sự rõ theo epoch là QLIKE (0.565→0.556→0.546), gợi ý train dài hơn có lợi cho
  chất lượng dự báo phương sai dù không giúp DirAcc.

**Kết luận Phần B (không đổi so với báo cáo 07-18, cập nhật thêm 1 baseline mới):** sau **10
biến thể** thử nghiệm tích hợp tin tức (khác nhau về nguồn dữ liệu, cơ chế fusion, loss phụ),
**chưa biến thể nào vượt được HAR-only trên DirAcc** — chỉ báo hiệu tín hiệu tin tức tiếng Việt
hiện có (thưa, coverage thấp) chưa đủ mạnh để cải thiện dự báo hướng biến động so với chỉ dùng
đặc trưng giá (HAR). Một số biến thể (Gated Cross-Attn, REST-TS, Dual-group+EWMA) đạt điểm tốt
hơn trên R²/QLIKE — cho thấy cơ chế fusion/loss SOTA hoặc feature set phong phú hơn CÓ cải thiện
chất lượng dự báo theo hướng khác DirAcc, dù chưa đủ để soán ngôi ở chỉ số chính.

---

## Phần C — Baseline khác (chưa có bộ 6-metric tổng hợp đầy đủ để so sánh trực tiếp)

Các hướng sau đã được triển khai/review kỹ nhưng KHÔNG có file kết quả tổng hợp 6-metric cuối
cùng trong repo tại thời điểm viết báo cáo này (không phải "không chạy được", mà là chưa có
kết quả "chính thức" để trích dẫn công bằng cùng bảng trên):

| Hướng | Trạng thái | Vị trí code | Lý do chưa đưa vào bảng so sánh |
|---|---|---|---|
| **TimesFM 2.5 (LoRA fine-tune)** | Implementation hoàn chỉnh, qua 3 vòng adversarial review (40 bug fixed), 34/34 test pass | `src/timesfm_baseline/` | Docs hiện có (`docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md`, `docs/TIMESFM_ADVERSARIAL_REVIEW_SUMMARY.md`) tập trung vào review/bug-fix, không thấy file kết quả train cuối cùng (`results.json` tương ứng không tồn tại) |
| **LSTM-HAR-GAT Hybrid** | Có chạy training (`results/lstm_har_gat_hybrid_2026-06-20_*/training_results.json`) | `src/lstm_har_gat_hybrid/` | Chỉ 2 epoch mỗi run trong log tìm được — không đủ để coi là kết quả cuối cùng, cần xác nhận lại trước khi trích dẫn |
| **Sentiment KNN baseline** | Có checkpoint model (`best_parallel_model.pth`) | `src/sentiment_baseline/` | Không tìm thấy file metrics/results.json đi kèm trong các thư mục `results/sentiment_baseline_knn_*/` đã kiểm tra |
| **CryptoMamba (enhanced/v2)** | Nhiều run (~25 folder kết quả) | `src/cryptomamba_baseline/` | Đây là thử nghiệm kiến trúc trên dữ liệu khác (không phải VN30 volatility mainline theo tên gọi) — cần làm rõ phạm vi áp dụng trước khi đưa vào so sánh chính |
| **TimesNet baseline** | 8 run (06-20) | `src/timesnet_baseline/` | Tương tự — chưa thấy results.json tổng hợp cuối cùng ở các thư mục đã kiểm tra |

**Khuyến nghị:** nếu thầy cần số liệu đầy đủ cho các hướng này, cần 1 lượt chạy lại
(re-run + trích xuất metrics) riêng — không nằm trong phạm vi báo cáo so sánh này vì rủi ro trích
dẫn số liệu chưa "chốt".

---

## Kết luận tổng thể

1. **Kiến trúc tốt nhất tổng thể (không tin tức):** Parallel LSTM-GNN (k-NN) — 69.61-69.98%
   DirAcc tuỳ run, R² 0.711, vượt xa các model tuyến tính/LSTM đơn giản trên DirAcc & R², đánh
   đổi bằng RMSE/MSE cao hơn (do bản chất dự báo đa dạng thay vì gần-hằng-số).
2. **Tích hợp tin tức: 10/10 biến thể đã thử đều CHƯA vượt được HAR-only trên DirAcc** — đây là
   kết quả nhất quán, lặp lại qua nhiều cách tiếp cận khác nhau (concat đơn giản, gate cứng, gate
   học được, cross-attention, alignment loss, residual supervision, dual-group+EWMA) → gợi ý vấn
   đề nằm ở CHẤT LƯỢNG/ĐỘ PHỦ tín hiệu tin tức tiếng Việt hiện có hơn là ở cơ chế fusion.
3. **Điểm sáng phụ:** một số biến thể tin tức đạt kỷ lục ở R² (Gated Cross-Attn, Dual-group+EWMA:
   0.716) và QLIKE (REST-TS: 0.543) — có giá trị nếu tiêu chí đánh giá chính của luận văn là các
   chỉ số này thay vì DirAcc.
4. **Baseline mới nhất (Dual-group+EWMA, 07-25)** không tạo đột phá so với 9 baseline tin tức
   trước — xếp giữa bảng trên DirAcc (68.71% ở 40 epoch, epoch-matched với Embedding-baseline
   gốc 68.76%@40ep — gần như ngang nhau), ngang hàng đầu trên R², và tốt hơn Embedding-baseline
   gốc trên QLIKE (0.546 vs 0.553). Đã kiểm tra hội tụ bằng 3 mức epoch (10/20/40, early-stop tại
   36) theo yêu cầu — xác nhận DirAcc đã bão hoà, không cải thiện thêm dù train dài hơn. Đóng góp
   chính của baseline này là phương pháp luận (dual-group source split + multi-window EWMA decay)
   và phát hiện + fix 1 bug rò rỉ dữ liệu (data leakage) thật trong quá trình implement, không
   phải điểm số đột phá.

## Phụ lục — Nguồn số liệu (để thầy kiểm tra lại)

- Phần A: `docs/report_2026-06-27/01_main_report/MODEL_COMPARISON_FINAL_REPORT.md`
- Phần B (9 baseline đầu): `docs/reports/2026-07-18_master_report_sota_news_fusion_baselines.md`
- Phần B (baseline mới): `docs/reports/2026-07-25_0131_summaryOfUpdate_report.md`,
  `results/dual_group_news_2026-07-25_011719/results.json` (10ep, đã fix leakage),
  `results/dual_group_news_2026-07-25_012212/results.json` (20ep, đã fix leakage),
  `results/dual_group_news_2026-07-25_071825/results.json` (40ep, early-stop ep36, đã fix leakage)
- Từng baseline có `requirements.md`/`design.md`/`code_review/*.md` riêng trong `baselines/<tên>/`
