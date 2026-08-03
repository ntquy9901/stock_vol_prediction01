# Báo cáo tổng hợp cuối cùng — Paper Readiness (2026-08-03)

Phạm vi: toàn bộ công việc audit/fix/verify thực hiện tự động trong phiên 2026-08-02 đêm →
2026-08-03, tiếp nối `docs/reports/2026-08-02_1056_paper_readiness_audit_report.md` và
`docs/reports/2026-08-02_1547_consolidated_fix_plan.md`. File này là điểm tham chiếu chính để
viết paper — các báo cáo trung gian khác (`2026-08-02_1527`, `2026-08-02_152253`,
`2026-08-02_152758`) giữ nguyên làm lịch sử, không cần đọc lại.

## 1. Tóm tắt kết quả cuối cùng

Sau khi sửa toàn bộ bug ảnh hưởng tới độ chính xác số liệu (mục 3), pipeline headline và pipeline
đối chứng HAR-only được train lại 1 lần, cùng 20 epoch, cùng seed=42, cùng ngày, trên cùng dữ liệu
đã sửa lỗi căn chỉnh ngày (P1.2):

| Metric | HAR-only backbone (ParallelLSTMGNN) | News-fusion (per-ticker gate) | Chênh lệch |
|---|---|---|---|
| QLIKE (test) | 0.4839 | **0.4641** | news-fusion tốt hơn 4.1% |
| RMSE (test) | 0.003025 | **0.002843** | news-fusion tốt hơn 6.0% |
| MAE (test) | 0.000826 | **0.000807** | news-fusion tốt hơn 2.3% |
| R² (test) | 0.7590 | **0.7873** | news-fusion tốt hơn |
| DirAcc — per-ticker, đúng (test) | **48.09%** | 47.56% | HAR-only nhỉnh hơn ~0.5pp (nhiễu) |
| DirAcc — flatten-biased, SAI, chỉ để đối chiếu | — | 71.64% | không dùng để kết luận |

Nguồn số liệu: `results/parallel_lstm_gnn_knn_2026-08-03_230722/training_results.json` (HAR-only),
`results/per_ticker_gate_2026-08-03_230821/results.json` (news-fusion).

**Kết luận có thể dùng cho paper:** tin tức (news-fusion, per-ticker gate) cải thiện các metric đo
sai số liên tục (QLIKE, RMSE, MAE, R²) so với HAR-only ở cùng điều kiện train, nhưng KHÔNG cải
thiện directional accuracy (chênh lệch ~0.5 điểm phần trăm, trong biên độ nhiễu 1 lần chạy, không
có ý nghĩa thống kê). Đây là **1 lần chạy** (seed=42), chưa phải xác nhận đa-seed — xem mục 6
"Giới hạn còn lại" trước khi viết thành kết luận cứng trong paper.

## 2. Bối cảnh: vì sao số liệu đêm nay khác các báo cáo trước

`docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` (báo cáo kiến trúc chính) đã gỡ DirAcc khỏi mọi bảng
từ "Cập nhật lần 3" do nghi ngờ công thức sai (xem `DIRACC_ISSUE_NOTE.md`). Phiên làm việc trước đó
(2026-08-02) cũng có 1 kết quả 5-seed epoch-20 cho per-ticker-gate: **QLIKE trung bình
0.5530±0.0115** — số liệu đó được sinh ra TRƯỚC khi bug P1.2 (căn chỉnh ngày sai lệch giữa các mã,
do ngày niêm yết khác nhau + xoá outlier độc lập theo từng mã) được sửa (commit `6672ffa`, cùng
ngày nhưng sau các lần chạy 5-seed đó). Số liệu 0.4641 trong mục 1 ở trên được sinh ra SAU khi P1.2
đã được sửa — do đó không so sánh trực tiếp được với 0.5530±0.0115; số 0.5530±0.0115 coi như đã lỗi
thời (superseded), không dùng trong paper.

REST-TS (`2026-07-18_resttext_baseline`, QLIKE 0.5431 tại thời điểm đó) đã được archive trong phiên
làm việc trước (quyết định của user: bỏ khỏi phạm vi paper, ưu tiên nhánh per-ticker-gate) — không
còn là điểm so sánh sống, chỉ còn giá trị lịch sử.

## 3. Toàn bộ bug đã audit + sửa trong phiên này (2026-08-02 đêm → 2026-08-03)

Commit range: `9332cd0..ef14da1` (10 commit, xem `git log --oneline 9332cd0..HEAD`).

### 3.1 P1.1 — Normalizer leakage + normalizer chưa từng được áp dụng (`src/lstm_gat_hybrid/dataset.py`)

Bug nghiêm trọng nhất phát hiện đêm nay, ảnh hưởng trực tiếp tới HAR-only backbone (kiến trúc
`ParallelLSTMGNN`, dùng làm đối chứng chính trong mọi bảng so sánh của báo cáo kiến trúc):

- **Leakage:** `MultiStockDataset.__init__` generate HAR feature + fit normalizer trên TOÀN BỘ dữ
  liệu (chưa split) trước khi `create_multi_stock_dataloaders` cắt theo vị trí bằng
  `torch.utils.data.Subset`.
- **Bug nặng hơn:** normalizer được fit nhưng KHÔNG BAO GIỜ được gọi `.transform()` trong
  `__getitem__` — nghĩa là model từ trước tới nay train trên dữ liệu HAR feature và
  `parkinson_volatility` HOÀN TOÀN THÔ (chưa chuẩn hóa, scale ~1e-3, khác nhau giữa các mã). Đây
  đúng loại lỗi đã ghi nhận trong CLAUDE.md §"LSTM-GNN Normalization Failure (2026-06-21)" — fit
  scaler nhưng quên áp dụng — vẫn còn tồn tại ở `train.py`/`train_parallel.py` cho tới đêm nay.
- **Fix:** tái sử dụng lại đúng pattern split-first đã được kiểm chứng ở
  `dataset_with_graph_method.py` (load raw → split raw theo ngày → generate HAR riêng từng split →
  fit normalizer CHỈ trên train, copy — không fit lại — object đã fit sang val/test).
  `__getitem__` giờ áp dụng `.transform()` thật. `validate()` trong `train.py`/`train_parallel.py`
  được thêm bước inverse-transform về scale gốc trước khi tính 6 metric bắt buộc.
- Một hack `loss = loss * 10.0` trong `train.py` (bù trừ cho việc train trên scale thô) đã được gỡ
  bỏ vì không còn cần thiết khi normalization thật đã hoạt động.
- Verify: 22/22 test trong `tests/lstm_gat_hybrid/` pass, smoke run 2 epoch xác nhận loss về scale
  O(1) (chuẩn hóa đúng), metric quay lại scale volatility gốc (~1e-3), prediction không còn hằng số.
- Commit: `1e5b551`, review fix: `947635b`.

### 3.2 P1.2 — Căn chỉnh ngày lệch giữa các mã (đã sửa trong phiên trước 2026-08-02, xác nhận lại đêm nay)

`dataset_with_graph_method.py`: outlier removal độc lập theo từng mã (drop hàng khác nhau) +
stacking theo VỊ TRÍ khiến vị trí `i` không còn nghĩa là cùng 1 ngày giao dịch giữa các mã một khi
các mã có ngày niêm yết khác nhau (chênh tới ~15 năm) hoặc có gap giao dịch khác nhau. Fix: winsorize
thay vì drop hàng (giữ nguyên số dòng), + `_reindex_to_common_dates()` ép mọi mã về đúng 1 trục ngày
chung trước khi index theo vị trí. 9 test trong `test_date_alignment_fix.py` pass. Commit gốc:
`6672ffa` (trước phiên này); tác động số liệu của fix này được định lượng gián tiếp ở mục 2 (QLIKE
per-ticker-gate cải thiện từ ~0.553 xuống ~0.464 sau khi fix có hiệu lực).

### 3.3 P1.3 — DirAcc flatten-bias chưa sửa hết ở 3 file HAR-only gốc

Công thức DirAcc đúng (per-ticker, tham số `n_stocks=`) đã áp dụng cho 22 baseline news-fusion ở
phiên trước, nhưng bỏ sót `src/lstm_gat_hybrid/{train,train_parallel,train_parallel_enhanced}.py`.
Đã bổ sung `n_stocks=` cho cả 3 file. Verify bằng smoke run thật (`train_parallel_enhanced.py`,
5 epoch): DirAcc đúng = 48.14% so với DirAcc sai (flatten-biased) = 71.57% — lệch 23.4 điểm phần
trăm, đúng như pattern đã ghi nhận (lệch 20-40pp) ở các baseline khác. Test:
`tests/lstm_gat_hybrid/test_diracc_per_ticker_fix.py` (8 test, bao gồm test tích hợp qua
`validate()` thật, không chỉ test hàm thuần). Commit: `45a364c`, `8ae79d0`, bổ sung test nhánh
denormalize: `947635b`.

**Đã kiểm tra thêm 3 nghi vấn khác** (sanity_constant_baseline.py, timesnet_baseline/train.py,
lstm_har_gat_hybrid/train_hybrid.py): 2/3 không phải bug thật (1 file cố ý báo cáo song song cả 2
công thức để đối chiếu; 1 file train từng mã riêng lẻ nên không có lỗi flatten). File còn lại
(`train_hybrid.py`) có bug thật nhưng không có số liệu nào của nó được trích dẫn cho paper (báo cáo
trước đã ghi "chỉ 2 epoch, chưa đủ để trích dẫn") và phụ thuộc `torch_geometric` (không cài trong
môi trường hiện tại) — để lại như follow-up, không sửa.

### 3.4 AUD-016 — Guard giá trị non-finite trong metric

`src/common/evaluation.py::assert_finite_metrics` — raise `ValueError` nêu rõ tên metric nếu
NaN/Inf lọt vào kết quả (thay vì âm thầm ghi vào `results.json`). Gọi ở cuối `evaluate_predictions()`
nên áp dụng tự động cho mọi caller. 7 test mới, không phá vỡ 145 test hiện có ở các baseline dùng
hàm này. Commit: `361902f`.

### 3.5 AUD-018 — Hardening `temporal_split.py`

Thêm validate: raise nếu có ngày NaT/không parse được, ngày trùng lặp (đã grep xác nhận mọi caller
hiện tại đều dùng dữ liệu 1-dòng-1-ngày-1-mã nên trùng lặp là lỗi thật, không phải use-case hợp lệ),
split rỗng (dataset quá nhỏ so với tỷ lệ). 8 test mới. Commit: `f1654a2`.

### 3.6 Điều tra 2 cặp giá trị trùng lặp bất thường (audit gốc §1.8)

2 cặp giá trị DirAcc giống nhau tới 11-13 chữ số thập phân giữa các run tưởng như độc lập, không
seed — nghi ngờ ban đầu là cache/checkpoint reuse. Điều tra thực tế (so sánh từng field trong
`results.json`, decompose tỷ lệ match thành k/N nguyên): **không phải cache/checkpoint bug** — MSE,
RMSE, R², QLIKE khác nhau giữa 2 run trong mỗi cặp, chỉ riêng DirAcc (công thức flatten-biased) là
giống nhau, vì công thức đó bị chi phối bởi cấu trúc chênh lệch mức volatility ổn định GIỮA CÁC MÃ
(vd blue-chip vs small-cap) — bất kỳ model nào train trên cùng dữ liệu HAR thật đều tái tạo gần
giống hệt số liệu này bất kể seed. Cả 2 pipeline liên quan tới cặp này đều đã archive (không cần
sửa code). Kết luận đã ghi vào audit report gốc. Commit: `a588bb0`.

### 3.7 Đóng `objective_news_baseline` (quyết định, không phải bug)

Baseline dừng ở bước extraction (đã đạt go-criterion riêng: 341 record, 113 stock-day test-period),
chưa từng train. Quyết định: đóng, không tiếp tục, ưu tiên thời gian cho việc verify bug + train
lại pipeline chính trước hạn nộp. Không archive folder (khác "null result", đây là "chưa hoàn
thành do ưu tiên"). Lý do đầy đủ:
`baselines/2026-07-15_objective_news_baseline/requirements/requirements.md` §7. Commit: `2914352`.

## 4. Code review (adversarial, subagent độc lập)

Review toàn bộ diff `6e5a618..1e5b551` (P1.1 fix, phần lớn thay đổi trong đêm). Kết quả: **0
Critical, 2 Important, 4 Minor.**

- Important #1 (nhánh denormalize trong `train.py`/`train_parallel.py` chưa có test) — đã fix,
  thêm test tích hợp, verify pass (commit `947635b`).
- Important #2 (`min_delta=1e-6` không còn phù hợp scale loss đã chuẩn hóa, early stopping gần như
  vô hiệu) — đã fix, tăng lên `1e-4`, ghi chú rõ scale thay đổi (commit `947635b`).
- 4 Minor (không block): finite-guard có thể abort run dài nếu 1 epoch validation ra giá trị
  non-finite tạm thời (chấp nhận được, đúng ý đồ AUD-016); path `data_dir=` cũ trong
  `MultiStockDataset` vẫn giữ leakage cũ (không dùng cho train thật, chỉ 1 script ad hoc); vài
  script debug (`sanity_constant_baseline.py`, `check_vhm_normalizer.py`) có thể cần cập nhật theo
  return-shape mới, chưa kiểm tra (follow-up); `VolatilityNormalizer.fit` dùng 1 scalar mean/std
  chung cho cả 3 feature HAR (tự nhất quán, chỉ ghi chú để biết đây là lựa chọn có chủ đích).

**Đánh giá cuối của reviewer:** "Ready to merge: With fixes" — cả 2 Important đã fix và verify.

## 5. Evidence capture (Gates 1-6, `verify-audit-fixes` skill)

`docs/reports/evidence/2026-08-03_0100/` (commit `ef14da1`). Gate 1 (repo identity): dirty tại thời
điểm chạy (results/ chưa track — bình thường). Gate 2 (static checks): **fail** — 947 lỗi ruff toàn
repo, đã xác nhận là nợ kỹ thuật có từ trước, KHÔNG phải do diff đêm nay (không đổi so với lần đo
trước). Gate 3 (test discovery): **fail** — đúng 9 lỗi collection đã biết từ trước (thiếu
`torch_geometric`/`mlflow`, import module đã archive trong `tests/` gốc) — cùng danh sách file với
lần đo trong archive-batch log, không có lỗi mới. Gate 4/5 (full test/smoke): **fail** — cascade từ
cùng 9 lỗi collection ở Gate 3 (pytest dừng thu thập toàn bộ khi có lỗi collection), không phải lỗi
mới trong code đã sửa. Gate 6 (coverage): **not_run** — cùng nguyên nhân cascade.

**Verify thay thế cho phần code thực sự thay đổi đêm nay** (không phụ thuộc vào 9 lỗi collection
kể trên, vì các thư mục bị lỗi không liên quan): `python -m pytest tests/lstm_gat_hybrid/
tests/common/ -q` → **38/38 pass** (chạy trực tiếp sau mỗi commit, xem log các commit `1e5b551`,
`361902f`, `f1654a2`, `947635b`).

**`diff-cover --fail-under=100` (C0 gate):** vẫn chưa cài đặt trong môi trường này (đã ghi nhận là
tooling gap trong CLAUDE.md từ trước) — không claim đã đạt C0=100%, ghi `Not run` theo đúng yêu cầu
CLAUDE.md khi chưa cài tool.

## 6. Giới hạn còn lại (chưa làm, cần biết trước khi viết paper)

1. **Chỉ 1 seed (42) cho bảng so sánh cuối cùng ở mục 1.** Chưa có xác nhận đa-seed (mean±std,
   kiểm định ý nghĩa thống kê) cho số liệu post-fix. Kết luận "QLIKE/RMSE/MAE/R² tốt hơn" nên được
   diễn đạt là kết quả 1 lần chạy, không phải kết luận đã kiểm định thống kê, trừ khi chạy thêm
   multi-seed trước khi nộp.
2. ~~VN30 universe staleness — vẫn chưa đóng băng chính thức~~ — **[SỬA LẠI, 2026-08-03]**: kiểm tra
   lại thấy quyết định này ĐàXONG từ 2026-08-02, không phải còn treo như ghi nhầm ở bản trước của
   report này. Xem `docs/reports/2026-08-02_1634_vn30_data_source_audit.md` §5-7: universe đã chốt
   **33 mã** (28/30 mã VN30 chính thức hiệu lực đến 2026-08-02 + 5 mã ngoài danh sách giữ lại vì đủ
   dữ liệu lịch sử — BCM, BVH, NVL, PDR, POW; loại BSR/VPL vì lịch sử giao dịch quá ngắn, có định
   lượng cụ thể). Xác nhận lại 2026-08-03: `data/processed/` hiện có đúng 33 file ticker (`ls
   data/processed/*_processed.csv | wc -l` → 33), khớp chính xác bảng đã chốt. Đoạn văn dùng cho
   Limitations/Dataset Description của paper đã có sẵn ở §7 của báo cáo đó (BSR/VPL: số phiên giao
   dịch cụ thể, lý do loại). Không cần làm gì thêm — chỉ cần paper trích dẫn đúng file đó.
3. **947 lỗi ruff + 9 lỗi pytest collection** — nợ kỹ thuật toàn repo, không chặn số liệu paper
   nhưng nên biết nếu công khai repo cùng paper.
4. **`sanity_constant_baseline.py`, `check_vhm_normalizer.py`, `debug_corrupted_val_batches.py`**
   — dùng `create_multi_stock_dataloaders` đã đổi return shape (dataset đầy đủ thay vì `Subset`) —
   chưa kiểm tra các script debug này còn chạy đúng không (không phải pipeline paper, chỉ là công
   cụ debug — follow-up, không urgent).
5. **`src/lstm_har_gat_hybrid/train_hybrid.py`** vẫn còn bug DirAcc flatten-bias, không sửa (không
   cited, phụ thuộc `torch_geometric` không cài) — xem mục 3.3.
6. **Canonical results table đầy đủ cho MỌI baseline còn giữ lại** (không chỉ HAR-only vs
   news-fusion) chưa được dựng — nếu paper cần bảng so sánh tất cả baseline (dual-group, spillover,
   calendar-gate, horizon 1/10/22...), cần 1 lượt inference-only re-evaluation riêng dùng
   checkpoint đã lưu, áp công thức DirAcc đã sửa — chưa làm trong phiên này do ưu tiên thời gian
   cho headline comparison ở mục 1.

## 7. Definition of Done — checklist

- [x] Code sửa đúng scope, không refactor thừa (surgical, xác nhận qua code review độc lập)
- [x] Test cho mọi behavior change (38 test lstm_gat_hybrid+common đêm nay, tổng cộng pass)
- [x] Checks chạy thật: pytest (38/38 khu vực liên quan), ruff (đã chạy, biết nợ kỹ thuật cũ)
- [x] Code review (adversarial, subagent riêng) — 0 Critical, 2 Important đã fix, 4 Minor ghi nhận
- [x] Archive/ không bị đụng tới trong audit/review (tuân thủ scope exclusion)
- [x] Push remote sau mỗi task — toàn bộ 10 commit đã push `origin master`, xác nhận không lệch
      (`git log origin/master..HEAD` rỗng tại mỗi bước)
- [x] Smoke test thật (không chỉ unit test thuần): 2 lần train thật 20-epoch (HAR-only,
      news-fusion) hoàn tất, kết quả ở mục 1
- [x] Impact analysis: grep xác nhận scope ảnh hưởng trước khi sửa (P1.1 chỉ ảnh hưởng
      train.py/train_parallel.py, không đụng news-fusion lineage đã fix riêng)
- [x] Similar-check: grep toàn repo cho pattern DirAcc-flatten tương tự, xử lý hoặc ghi follow-up
      rõ ràng (mục 3.3)
- [x] Summary report này — đúng văn phong khách quan theo CLAUDE.md (không xưng hô cá nhân, không
      tự nhận "trung thực", chỉ nêu sự kiện/số liệu/nguồn)

## 8. Việc tiếp theo đề xuất (không tự làm thêm trong phiên này, cần quyết định người dùng)

- Multi-seed rerun cho bảng so sánh mục 1 nếu paper cần kết luận có ý nghĩa thống kê.
- Đóng băng chính thức VN30 universe hoặc xác nhận giữ nguyên + ghi Limitations.
- Canonical table đầy đủ mọi baseline (mục 6, điểm 6) nếu paper cần so sánh > 2 kiến trúc.
- Cài `diff-cover`/`ruff` vào quy trình CI chính thức nếu muốn gate C0=100% thực thi được máy móc.
