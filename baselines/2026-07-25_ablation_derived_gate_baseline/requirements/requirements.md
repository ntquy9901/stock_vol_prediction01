# Requirements (Specify) — Ablation-Derived Gate Baseline

**Baseline:** `2026-07-25_ablation_derived_gate_baseline` · Theo SDD (CLAUDE.md §1.5).
**Nguồn:** `2026-07-25_news_usefulness_ablation` (đo trực tiếp trên LSTM-GNN, epoch-matched
10-vs-10, QLIKE làm tiêu chí chính — xem code_review của baseline đó về bug epoch-mismatch đã
fix trước khi dùng danh sách này).

## 1. Danh sách ON/OFF

**NEWS-ON (11 mã):** HDB, HPG, MWG, NVL, PDR, PLX, SSI, VHM, VJC, VPB, VRE
**NEWS-OFF (21 mã):** ACB, BCM, BID, BVH, CTG, FPT, GAS, GVR, MBB, MSN, POW, SAB, SHB, SSB, STB,
TCB, TPB, VCB, VIB, VIC, VNM

## 2. Kiến trúc

Tái dùng 100% `SelectiveGateNewsBaseline` (sibling `2026-07-25_selective_news_gate_baseline`) —
chỉ đổi ticker set, giống pattern `2026-07-25_top3_news_gate_baseline`.

## 3. Success criteria / diễn giải

So sánh Test DirAcc + QLIKE với:
- All-ON (không mask, 10ep): 68.50% DirAcc, QLIKE 0.565
- 22-mã gate (EDA rộng, đã bác bỏ): 67.56% DirAcc
- 3-mã gate (EDA hẹp, hòa): 68.23% DirAcc
- HAR-only: 69.98% DirAcc
- HAR-only reference (epoch-matched, từ chính ablation): 68.42% DirAcc, QLIKE 0.562

Vì danh sách này được chọn DỰA TRÊN QLIKE cải thiện (không phải DirAcc), kỳ vọng hợp lý nhất là
**QLIKE tổng thể cải thiện rõ hơn DirAcc** — nếu đúng, đây là bằng chứng phương pháp "tự đo bằng
chính kiến trúc" tốt hơn "mượn từ model khác", dù chỉ ở 1 metric.

## 4. Lưu ý

Single-seed (đã ghi trong baseline ablation) — kết quả có thể vẫn nhiễu. Không train quá 10 epoch
nếu chưa có xác nhận thêm.
