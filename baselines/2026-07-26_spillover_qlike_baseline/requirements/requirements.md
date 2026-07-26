# Requirements (Specify) — Directed Volatility-Spillover Graph + QLIKE-Augmented Loss

**Baseline:** `2026-07-26_spillover_qlike_baseline` · Theo SDD (CLAUDE.md §1.5).
**Depends on:** `2026-07-25_dual_group_news_embedding_baseline` (reuses its panel/dataset/model
read-only), rebuilt panel from tonight's 12-new-source dual-group retrain (task #1/#2 this session).

## 1. Bối cảnh

Toàn bộ các baseline thử trước đây (dual-group, macro, gated cross-attn, selective gate, top3
gate, ablation) đều giữ NGUYÊN 2 thứ: (a) đồ thị liên-cổ-phiếu (stock graph) — luôn là đồ thị
**đối xứng, cùng-ngày (contemporaneous)** dựa trên tương quan Pearson (`graph_correlation.py`:
`construct_correlation_graph` threshold hoặc `construct_knn_graph` top-k, cả 2 đều symmetrize
`adj[i,j]=adj[j,i]`); (b) loss huấn luyện = MSE thuần trên scale đã chuẩn hoá. Chỉ phần NEWS
BRANCH được thay đổi qua các baseline — và tất cả đều cho kết quả null (không thắng rõ HAR-only).

Sau khi user yêu cầu (2026-07-26) tự nghiên cứu SOTA và đề xuất hướng cải thiện mới (không cần
chờ approve), research 2 nhóm paper 2025-2026 (xem `design.md` §1 để có nguồn) chỉ ra đúng 2 điểm
yếu này:

1. **Đồ thị nên có hướng (directed), không đối xứng** — Zhang, Pu, Cucuringu & Dong (IJF 2025,
   "Forecasting realized volatility with spillover effects: Perspectives from GNN") và Chi et al.
   (J. Forecasting 2026) đều dùng đồ thị lan truyền cú sốc biến động (volatility spillover) có
   hướng (ai truyền cú sốc cho ai), khác hẳn đồ thị tương quan đối xứng hiện tại của project.
2. **QLIKE nên là (một phần) hàm loss huấn luyện, không chỉ là eval metric** — cùng paper trên báo
   cáo training với QLIKE loss cải thiện đáng kể so với MSE thuần.

Đây là hướng **KHÁC HẲN** các thử nghiệm trước (không phải thêm 1 biến thể news-fusion khác) — sửa
đúng 2 thành phần literature-backed mà project chưa từng đổi.

## 2. Giả thuyết

1. Đồ thị lan truyền biến động **có hướng** (directed lead-lag: cú sốc biến động ngày `t` của mã
   A dự báo biến động ngày `t+1` của mã B) nắm bắt quan hệ nhân-quả-thời-gian mà đồ thị tương quan
   cùng-ngày, đối xứng bỏ lỡ → cải thiện dự báo, đặc biệt Dir Acc/QLIKE.
2. Thêm thành phần QLIKE (trên scale gốc, đã inverse-transform, clamp dương) vào loss huấn luyện
   (bên cạnh MSE) giúp model tối ưu trực tiếp theo metric học thuật chuẩn cho volatility → giảm
   QLIKE test, có thể cải thiện R²/Dir Acc.

## 3. Mục tiêu / Output

1. `graph_spillover.py`: `construct_directed_spillover_graph(volatility_window, k=8)` — đồ thị
   **có hướng**, cạnh `i <- j` (i nhận, j truyền) trọng số = |corr(vol_j[t], vol_i[t+1])| (lag-1
   cross-correlation), giữ top-k cạnh vào mạnh nhất mỗi node `i` (không symmetrize). Cùng chi phí
   tính toán O(num_stocks²) như đồ thị hiện có (không thêm nguồn dữ liệu mới).
2. `dataset_spillover_news.py`: copy có sửa của `dataset_dual_news.py` (hard isolation, CLAUDE.md
   §3.F rule 3) — thêm `graph_method='spillover'` gọi hàm graph mới; mọi thứ khác (HAR cols, news
   panel loader, sequence windowing) giữ nguyên logic sibling.
3. `train_spillover_qlike.py`: copy có sửa của `train_dual_news.py` — loss = `MSE_norm + λ·QLIKE_clamped`
   (λ mặc định 0.1, có thể chỉnh qua `--qlike_weight`), model KHÔNG đổi
   (tái dùng `DualGroupNewsBaseline`), dùng dataset mới ở trên. Cap 10 epoch (enforced in code).
4. Panel input: dùng `dual_group_news_panel.parquet` đã rebuild (12 nguồn mới) từ task #1 phiên
   này — không build panel riêng.

## 4. Cô lập (hard isolation, CLAUDE.md §3.F rule 3)

Import read-only từ `2026-07-25_dual_group_news_embedding_baseline`
(`model_dual_news.DualGroupNewsBaseline`, `dataset_dual_news.load_news_panel`, `HAR_COLS`,
`_norm_date`) và từ `src.lstm_gat_hybrid` (`config`, `train_parallel_enhanced.EarlyStopping` +
`plot_learning_curves_with_analysis`). KHÔNG sửa file nào của baseline khác hay của
`src/lstm_gat_hybrid`. `graph_correlation.py` giữ nguyên — đồ thị mới sống trong code/ riêng.

## 5. Success criteria / Go-No-go

- [x] `graph_spillover.py`: đồ thị output **không đối xứng** (asymmetric) trên dữ liệu tương quan
      lệch pha thật (property test) + xử lý an toàn zero-variance/degenerate windows.
- [x] Dataset shapes đúng (giống sibling, chỉ khác adjacency construction).
- [x] Combined loss (MSE + QLIKE clamp) không NaN/Inf trong 10 epoch thật.
- [x] Train 10 epoch thật, in đủ 6 metrics mỗi 5 epoch + val/test comparison.
- [x] So sánh với dual-group baseline rebuilt (task #2 phiên này) và với record hiện có
      (gated-crossattn R²=0.7157/QLIKE=0.557, HAR-only DirAcc=69.98%) — không kỳ vọng thắng chắc,
      đây là baseline THỬ hướng mới; kết quả null vẫn có giá trị (loại trừ hướng graph/loss).

  **KẾT QUẢ THẬT (2026-07-26, 10 epoch, panel rebuilt 12-nguồn-mới):**
  | Metric | Dual-group (symmetric graph + MSE) | Spillover+QLIKE (directed graph + MSE+QLIKE) | Diff |
  |---|---|---|---|
  | Test DirAcc | 68.25% | 68.23% | -0.02pp |
  | Test R² | 0.7124 | 0.7132 | +0.0008 |
  | Test QLIKE | 0.5598 | 0.5622 | +0.0024 (worse) |
  | Test RMSE | 0.002651 | 0.002647 | -0.000004 (better, negligible) |

  **KẾT LUẬN: NULL — không cải thiện rõ rệt.** Đồ thị có hướng (lag-1 lead-lag) + QLIKE-augmented
  loss cho kết quả GẦN NHƯ GIỐNG HỆT baseline đối chứng cùng panel (chênh lệch nằm trong nhiễu
  train-to-train, không phải cải thiện có ý nghĩa). Không thắng HAR-only (69.98% DirAcc) hay
  gated-crossattn record (R²=0.7157/QLIKE=0.557). Đây là null result thứ ~10 liên tiếp trong
  project (dual-group, macro, gated cross-attn, selective/top3 gate, ablation, REST-TS, alignment
  loss, pure market, market fallback, latent noise, và giờ spillover+qlike) — bằng chứng ngày càng
  mạnh rằng nút thắt KHÔNG nằm ở graph construction hay loss function, mà có thể ở chỗ khác (xem
  summary report `docs/reports/2026-07-26_*_summaryOfUpdate_report.md` phần Risks/follow-ups).
- [x] pytest pass (20/20 — shape + property tests cho graph + loss), self-adversarial code review
      chạy trước khi coi "done" (user vắng mặt, không chờ `/code-review` checkpoint — theo tiền lệ
      macro baseline tối qua). Xem `code_review/code_review_2026-07-26.md`.

## 6. Out of scope

- Không dùng Diebold-Yilmaz spillover index đầy đủ (variance decomposition of VAR forecast error)
  — quá phức tạp cho scope 1 baseline thử nghiệm (Simplicity Gate); dùng proxy đơn giản hơn (lag-1
  lead-lag cross-correlation, cùng họ ý tưởng "directed volatility transmission", chi phí tính
  toán tương đương đồ thị cũ).
- Không tune λ (qlike_weight) sâu — dùng 1 giá trị mặc định hợp lý (0.1), ghi rõ là chưa tune.
- Không kết hợp với news-fusion variant khác (macro, gated gate) — cô lập biến số thử nghiệm, chỉ
  đổi graph + loss trên nền dual-group panel đã có.
