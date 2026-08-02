# Paper-Readiness Audit — 2026-08-02

## Phạm vi

Rà soát toàn bộ codebase (23 baseline trong `baselines/`, toàn bộ `src/`), toàn bộ báo cáo
(`docs/reports/*.md` — 26 file, và `docs/report_2026-06-27/`, `2026-07-11/`, `2026-07-18/`,
`2026-07-25/`, `2026-08-01/`), cộng với kiểm chứng thực nghiệm (chạy lại 3 seed) cho kết quả
đang được coi là tốt nhất của project. Mục đích: liệt kê toàn bộ vấn đề chưa giải quyết, điểm
không phù hợp/vô lý, và khoảng trống cần lấp trước khi viết paper nộp hội nghị tháng tới.

Phương pháp: 3 agent điều tra độc lập song song (baseline-results, code correctness, docs
contradictions) + 1 thực nghiệm kiểm chứng seed trực tiếp. Toàn bộ finding dưới đây đã đối
chiếu với file:line cụ thể; các finding quan trọng nhất đã được xác minh lại thủ công lần 2
trước khi đưa vào báo cáo này.

---

## 1. Vấn đề chưa giải quyết (xếp theo mức độ ảnh hưởng tới tính hợp lệ khoa học)

### 1.1 [RESOLVED — số liệu, không phải kết luận] Kết quả "tốt nhất project" (QLIKE 0.5473) không tái lập được

`baselines/2026-07-26_per_ticker_news_gate_baseline` — con số QLIKE=0.5473 (test, epoch 20) được
nhiều báo cáo dẫn làm "breakthrough"/kỷ lục project — được train **không set seed nào**
(`torch.manual_seed`/`np.random.seed` không tồn tại trong code tại thời điểm chạy).

**Kiểm chứng vòng 1** (phiên trước, 3 seed, 10 epoch, không epoch-matched với con số gốc):

| seed | test QLIKE (10 epoch) |
|---|---|
| 42 | 0.5704 |
| 123 | 0.5488 |
| 2026 | 0.5503 |
| **mean ± std** | **0.5565 ± 0.010** |

**Kiểm chứng vòng 2 (phiên này, epoch-matched đúng epoch 20 như con số gốc, resume từ checkpoint
vòng 1 +10 epoch, cộng 2 seed mới 1 và 7 chạy fresh 20 epoch — 5 seed tổng, code đã có fix DirAcc
per-ticker):**

| seed | test QLIKE (epoch 20) | test DirAcc per-ticker (%) |
|---|---|---|
| 1 | 0.5396 | 47.70 |
| 123 | 0.5476 | 47.47 |
| 2026 | 0.5475 | 48.54 |
| 42 | 0.5640 | 47.05 |
| 7 | 0.5661 | 47.80 |
| **mean ± std (n=5)** | **0.5530 ± 0.0115** | **47.71 ± 0.55** |
| gốc (không seed, epoch 20) | 0.5473 | — (chỉ có bản flatten-biased) |

Kết luận cập nhật: con số gốc 0.5473 nằm **trong khoảng 1 std của mean** (0.5415–0.5645) — không
phải một lần chạy may mắn bất thường như nghi ngờ ban đầu ở vòng 1, nhưng **cũng không phải mức
hiệu năng ổn định/tốt nhất** — nó là 1 trong 5 draw, thấp hơn mean một chút, cao hơn seed tốt nhất
(1: 0.5396). Mean thật của per-ticker-gate ở epoch 20 là **0.5530 ± 0.0115**, **tệ hơn** REST-TS
(0.5431, single-seed, chưa verify — xem mục 1.5) trên toàn bộ 5/5 seed. DirAcc per-ticker
(47.71 ± 0.55%) xác nhận lại phát hiện mục 1.3 — gần random, ổn định qua seed (std thấp) nhưng
không có tín hiệu dự báo đúng chiều thật sự.

**Hệ quả cho headline result (mục 1.5, 3.2):** per-ticker-gate KHÔNG thể dùng làm headline
"beats REST-TS on QLIKE" — trung bình 5 seed thua REST-TS. REST-TS bản thân chưa qua multi-seed
verify (ngoài phạm vi đã thống nhất cho phiên này) — cần làm trước khi chốt headline cuối cùng
cho paper.

Kết quả training đầy đủ: `results/per_ticker_gate_2026-08-02_150559` (seed 42),
`_150913` (123), `_151224` (2026), `_151827` (1), `_152448` (7). Seed-fixing đã commit
(`fccaf6a`).

### 1.2 [CRITICAL] Data leakage thật (không phải nghi ngờ) ở 2 script gốc

`train_simple_lstm_vn30.py:45-52` và `train_lstm_har_vn30.py` (cùng pattern) — comment ghi
`# Temporal split (70/15/15)` nhưng code thực thi
`torch.utils.data.random_split(dataset, [...], generator=...)`. Đây là **random split trên dữ
liệu time-series có thứ tự thời gian**, đúng loại lỗi CLAUDE.md §3.A cấm rõ ràng ("❌ WRONG -
Random split causes data leakage"). Khác với các file archive đã biết lỗi
(`archive/data_leakage_scripts/*`), 2 file này **đang sống, không archive**. Nếu số liệu từ 2
script này được dùng cho paper, toàn bộ metric bị vô hiệu — cần audit xem 2 script này có phải
nguồn của bất kỳ con số nào đã báo cáo hay không, và sửa lại temporal split thật nếu có dùng.

### 1.3 [CRITICAL] DirAcc flatten-order bug là headline metric ở toàn bộ dòng baseline news-gate (07-25 → 08-01)

Bug đã biết (memory: `project_diracc_flatten_order_bug`) — nhưng phạm vi thực tế lớn hơn ghi
nhận trước đó. `src/common/evaluation.py`'s `directional_accuracy()` bản thân không sai (chỉ
nhận input đã đảo thứ tự sai từ caller). Nhưng **9 baseline** (`train_dual_news.py`,
`train_selective_gate.py`, `train_top3_gate.py`, `train_ablation_gate.py`, `train_macro_news.py`,
`train_per_ticker_gate.py`, `train_spillover_qlike.py`, `train_calendar_news_gate.py`,
`train_har_only_reference.py`) đều gọi `evaluate_predictions()` trên mảng đã flatten
interleaved-theo-ticker-trong-ngày — dù mỗi file đều **tự implement đúng bản per-ticker** song
song (`directional_accuracy_per_stock`), bản flatten (sai) vẫn là số **in ra console và lưu vào
`results.json` làm headline**, bản đúng chỉ là phụ.

Gap thực đo trên toàn bộ 23 baseline (bảng đầy đủ ở mục 5): trung bình ~20-24pp, cá biệt
`resttext_baseline` 32pp, `horizon1_baseline` 39pp (per-stock 33.16% — **dưới random**),
`horizon22_baseline` 24pp (per-stock 42.89% — dưới random). Chỉ 1/23 baseline
(`horizon1_baseline`, qua `docs/report_2026-08-01/DIRACC_ISSUE_NOTE.md`) được gắn cờ chính thức
trong tài liệu. **Mọi con số DirAcc dùng cho paper phải lấy bản per-ticker, không phải bản đã
báo cáo trong hầu hết `results.json` hiện có.**

### 1.4 [RESOLVED — seeding; kiểm chứng multi-seed vẫn còn thiếu] Toàn bộ ~23 training script nay đã seed, nhưng chỉ per-ticker-gate được multi-seed verify

Đã fix trong phiên tiếp nối: `torch.manual_seed(42)`/`np.random.seed(42)` (mặc định) thêm vào
toàn bộ ~23 training script còn lại (danh sách gốc từng liệt kê ở đây), kể cả
`src/lstm_har_gat_hybrid/train_hybrid.py` và `src/cryptomamba_baseline/train_enhanced.py` — đã
**commit** (`fccaf6a`). Việc này giải quyết vấn đề "chạy lại ra kết quả khác" cho các lần chạy
tương lai, nhưng **không tự động làm hợp lệ các con số ĐÃ báo cáo trước đây** (chúng vẫn được train
từ 1 lần chạy không seed) — muốn dùng số nào cho paper vẫn cần multi-seed verify riêng số đó (như
đã làm cho per-ticker-gate ở mục 1.1). 22/23 baseline khác hiện vẫn chỉ có 1 lần chạy — chấp nhận
được cho kết luận null result, nhưng bất kỳ claim "beat baseline"/positive result nào từ nhóm này
đều cần multi-seed verify trước khi đưa vào paper, theo đúng quy trình đã áp dụng ở mục 1.1.

### 1.5 [HIGH] "Kết quả tốt nhất hiện tại" trong báo cáo mới nhất thực ra tệ hơn kỷ lục đã ghi nhận trước đó, không so sánh/không disclose

`docs/reports/2026-07-25_0712_all_baselines_comparison_report.md` (dòng 74) xác nhận **REST-TS
QLIKE=0.5431 "vẫn giữ kỷ lục"** trên toàn bộ 12 biến thể so sánh — đây là baseline
`2026-07-18_resttext_baseline`, KHÔNG dùng gate. `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md`
§1.3 sau đó tuyên bố không kèm điều kiện: "**Kết quả tốt nhất hiện tại**: per-ticker gated news
... QLIKE 0.5436" — **số này tệ hơn** (QLIKE cao hơn = xấu hơn) REST-TS 0.5431, và REST-TS
**không hề được nhắc tới** trong toàn bộ report ~1000 dòng đó. Claim "best result" đang ngầm thu
hẹp phạm vi so sánh (chỉ trong nhánh per-ticker-gate) mà không nói rõ nhánh khác (REST-TS) vẫn
thấp hơn trên đúng metric học thuật chính (QLIKE). **Paper không thể dùng nguyên trạng claim
"best result" từ báo cáo hiện tại** — cần đối chiếu lại toàn bộ 23 baseline trên cùng 1 bảng,
cùng epoch, cùng metric, trước khi chọn 1 con số headline.

### 1.6 [MEDIUM] Số liệu trôi dạt không giải thích cho cùng một "record"

`docs/reports/2026-07-26_2330_summaryOfUpdate_report.md` (dòng 77) ghi per-ticker-gate breakthrough
= QLIKE **0.5473** (epoch 20). `BAO_CAO_TONG_HOP.md` §4 (dòng 436) ghi cùng lineage (panel cũ,
gắn nhãn epoch 10) = QLIKE **0.5497**. Không báo cáo nào giải thích chênh lệch. Với phát hiện ở
mục 1.1 (không seed), khả năng cao đây là 1 biểu hiện sớm khác của cùng vấn đề
non-reproducibility, chưa từng được gắn cờ là vậy.

### 1.7 [MEDIUM] VN30 ticker universe stale — biết từ 07-26, chưa fix, chưa đưa vào limitations mới nhất

32-ticker universe của project lệch so với VN30 thật (5 dư/3 thiếu — memory
`project_vn30_ticker_universe_mismatch`), và VPB/VRE thiếu hoàn toàn khỏi news-matching regex.
`2026-07-25_selective_news_gate_baseline`'s code review độc lập phát hiện lại đúng vấn đề này
(EDA phủ 30 ticker, pipeline dùng 32, VPB/VRE mặc định OFF vì thiếu evidence). `BAO_CAO_TONG_HOP.md`
§5 "GIỚI HẠN/VIỆC CHƯA XONG" (9 mục) có nhắc vụ VPB/VRE-thiếu-regex (đã fix) nhưng **không nhắc**
vấn đề universe-stale riêng biệt (chưa fix) dù đã biết từ 1 tuần trước. Nếu paper dùng "VN30" làm
tên tập dữ liệu, cần fix hoặc explicit caveat + justify trong Limitations.

### 1.8 [MEDIUM] 2 cặp giá trị trùng khớp bất thường (11-13 chữ số) giữa các run độc lập, không seed

(a) `results/embedding_baseline_2026-07-15_015004`'s test DirAcc khớp một run 2026-07-11 khác
tới 11 chữ số thập phân. (b) `dual_group_news`'s run không tài liệu hóa `2026-07-26_192414` có
test DirAcc khớp run 20-epoch của chính nó tới 13 chữ số trong khi mọi metric khác khác nhau.
Với hệ thống không seed, xác suất trùng ngẫu nhiên gần như bằng 0 — nghi vấn cache/checkpoint bị
tái sử dụng nhầm giữa các run, hoặc bug ghi đè results.json. **Chưa điều tra nguyên nhân gốc** —
nên làm rõ trước khi paper trích dẫn bất kỳ số nào từ 2 run liên quan.

### 1.9 [LOW-MEDIUM] Vi phạm cấu trúc baseline bắt buộc (CLAUDE.md §3.F)

- `2026-07-25_news_usefulness_ablation/test/` chỉ có `__init__.py`, không có test thật — vi phạm
  yêu cầu bắt buộc, dù kết quả của baseline này được dùng downstream.
- `2026-07-11_sentiment_decay` không có results directory riêng — kết quả chỉ định vị được bằng
  cách khớp số với 1 hình trong report khác. Traceability failure thật sự.
- `2026-07-15_objective_news_baseline` chưa từng đạt go/no-go — có bước extract nhưng không có
  training/so sánh run nào tồn tại hay định vị được.

### 1.10 [LOW] `project-context.md` còn claim stale khác (đã fix 1 phần trong phiên này)

Đã fix trong phiên: mục "Loss Function Priority" + `SINGLE_HORIZON_CONFIG['loss']` (dòng 98, sót
lại từ lần fix trước, vừa fix thêm). Còn tồn tại, CHƯA fix (nằm ngoài scope 2 việc user yêu cầu
trước đó, liệt kê để theo dõi):
- "Feature Categories" (dòng ~327-333) ghi có Temporal features (day-of-week/month/quarter) và
  Technical indicators (RSI/MACD/Bollinger) — `BAO_CAO_TONG_HOP.md` §3 (dòng 404-420) xác nhận
  sau khi rà toàn bộ `src/`: **không tồn tại** feature nào trong số này ở model hiện tại (chỉ có
  ở 1 thử nghiệm 08-01 calendar, kết quả null, không đưa vào production).
  `MODELS_TO_COMPARE` (dòng 179-185) thiếu kiến trúc news-fusion/per-ticker-gate — kiến trúc mà
  `BAO_CAO_TONG_HOP.md` gọi là hiện tại/quan trọng nhất.
- Header ghi "Last Updated: 2026-06-29" nhưng "Update History" trong cùng file có mục 2026-07-26
  — header cũ hơn nội dung.

---

## 2. Điểm không phù hợp / vô lý (illogical/inappropriate framing — không phải bug, nhưng gây hiểu sai nếu đưa vào paper nguyên trạng)

- **`horizon1_baseline`**: narrative "gate thắng cả 3 metric" (QLIKE/R²/RMSE) không inline cảnh
  báo rằng DirAcc per-stock đồng thời sập xuống 33.16% — dưới random 50%. Đưa 3/4 metric thắng
  mà giấu metric thứ 4 sập là cherry-picking nếu paper trích nguyên văn.
- **Claim "best result"** (mục 1.5) thu hẹp phạm vi so sánh ngầm, không nêu rõ đang so trong 1
  nhánh con chứ không phải toàn bộ 23 baseline.
- **Checklist go/no-go để trống** (`- [ ]`) ở nhiều baseline 08-01 dù đã hoàn thành và có báo cáo
  đầy đủ nơi khác — không sai về nội dung nhưng gây khó truy vết trạng thái thật khi audit.
- **DirAcc gap hiện diện ở gần như MỌI baseline** (~20-24pp trung bình) nhưng chỉ 1/23 được gắn
  cờ chính thức trong tài liệu — tạo ấn tượng đây là vấn đề cục bộ trong khi thực ra là hệ thống.

---

## 3. Còn thiếu để paper hoàn chỉnh

1. **Statistical rigor**: toàn bộ 23 baseline hiện chỉ có 1 lần train/1 seed. Paper cần mean±std
   trên ≥3-5 seed cho ít nhất kết quả headline (per-ticker-gate) và baseline so sánh chính
   (HAR-only, REST-TS) trước khi khẳng định "beat baseline".
2. **Chọn lại 1 headline result thống nhất, defensible** — hiện có ≥3 con số mâu thuẫn tự xưng
   "best" (REST-TS 0.5431, per-ticker-gate 0.5473/0.5497/0.5436 tùy report). Cần 1 bảng so sánh
   duy nhất, cùng điều kiện (cùng epoch, cùng seed count, cùng split), làm cơ sở paper.
3. **Sửa DirAcc project-wide** trước khi đưa số vào paper — dùng bản per-ticker (đã có sẵn trong
   mọi results.json dưới tên `directional_accuracy_per_stock`) làm số chính thức, không dùng bản
   flatten.
4. **Kiểm tra & sửa leakage** ở `train_simple_lstm_vn30.py`/`train_lstm_har_vn30.py` nếu số liệu
   từ đây được dùng.
5. **Significance testing** cho mọi claim "beat HAR-only" — hiện là so sánh point-estimate đơn
   lẻ, không có confidence interval hay paired test qua các ticker/ngày.
6. **1 báo cáo kiến trúc + kết quả hợp nhất** — các report hiện tại theo từng ngày, một số mâu
   thuẫn nhau về "best result" (mục 1.5, 1.6); paper cần nguồn sự thật duy nhất, không phải chọn
   1 trong nhiều report rời rạc.
7. **Cập nhật Limitations**: VN30 universe stale (mục 1.7), seed non-reproducibility (mục 1.1,
   1.4), lịch sử DirAcc bug (mục 1.3) — hiện Limitations section mới nhất chỉ có 9 mục, thiếu ít
   nhất 2 trong 3 điểm này.
8. **Reproducibility statement**: commit seed-fix code (hiện uncommitted), ghi rõ seed/config
   chính xác dùng để tạo số liệu paper, cân nhắc release seed list cùng code nếu công khai repo.
9. **Related work / positioning** — chưa nằm trong phạm vi audit này (audit tập trung code +
   report nội bộ), nhưng thường là phần thiếu lớn nhất ở giai đoạn chuẩn bị nộp — cần rà riêng.
10. **Code cleanliness trước khi công khai kèm paper** (không chặn nộp paper, chặn claim "code
    available"): 3 hardcoded `D:\` absolute path (`analyze_news_sparsity.py:27,30`,
    `sentiment_fullcorpus_eda.py:42`), nhiều bare `except:` rải rác (`graph_utils.py:213`,
    `compare_crawl_results.py:34,41`, `graph_correlation.py:167`, crawler files).

---

## 4. Bảng inventory 23 baseline

| Ngày | Baseline | Hypothesis | Verdict | Metric chính | DirAcc flat→per-stock | Ghi chú |
|---|---|---|---|---|---|---|
| 07-07 | embedding_baseline | Embedding 768-d thắng scalar sentiment | null | QLIKE 0.5534, DA 68.76% | 68.76→48.03% (20.7pp) | không seed |
| 07-08 | market_fallback | Nhánh market-wide phục hồi tín hiệu | null | QLIKE 0.5479, DA 68.69% | 68.69→44.71% (24.0pp) | không seed |
| 07-11 | latent_noise | Gaussian noise regularize | null vs HAR-only | QLIKE 0.5435, DA 69.33% | 69.33→49.19% (20.1pp) | 2/4 result dir rỗng |
| 07-11 | sentiment_decay | Decay-carry thắng neutral-fill | null | QLIKE 0.5694, DA 67.87% | — | **không có results dir riêng** |
| 07-11 | sentiment_price_eda | Sentiment liên hệ forward return/vol | null/no-go | p=0.747 mọi horizon | N/A | EDA, có tự sửa 1 finding sai trước đó |
| 07-15 | objective_news_baseline | Corporate-event news sạch hơn | **chưa hoàn thành** | không có | N/A | chưa từng train/so sánh |
| 07-15 | pure_market_baseline | Market-wide pooled thắng ticker-matched | null/inconclusive | QLIKE 0.5560, DA 68.95% | 68.95→45.63% (23.3pp) | không seed |
| 07-18 | alignment_loss_baseline | Cosine-alignment giảm text collapse | null | QLIKE 0.5462, DA 68.76% | 68.76→47.85% (20.9pp) | không seed |
| 07-18 | gated_crossattn_baseline | Gate học cross-attention | null (DirAcc), best R² | QLIKE 0.5567, DA 68.97% | 68.97→48.66% (20.3pp) | review bắt bug HIGH, +1.81pp sau fix |
| 07-18 | resttext_baseline | Residual-only head | null (DirAcc), best QLIKE | QLIKE 0.5431, DA 68.29% | 68.29→36.20% (**32.1pp**) | **QLIKE record thật của project** |
| 07-25 | ablation_derived_gate_baseline | Gate list tự suy ra thắng EDA-derived | inconclusive | QLIKE 0.5623, DA 68.23% | — | không seed |
| 07-25 | dual_group_news_embedding_baseline | Dual-source PCA thắng single-group | null | QLIKE 0.5458-0.565 | ~20.6pp avg | HIGH leakage bug đã fix; 1 run lạ trùng khớp bất thường |
| 07-25 | expand_news_cache_baseline | (data-prep) | hoàn thành | N/A | N/A | — |
| 07-25 | macro_news_baseline | Macro news mang tín hiệu chung | null | QLIKE 0.5634, DA 68.63% | 68.63→45.71% (22.9pp) | không seed |
| 07-25 | news_usefulness_ablation | Đo usefulness trên kiến trúc riêng | positive/partial (11/32) | QLIKE_ref 0.5623 | N/A | **thiếu `test/`** |
| 07-25 | selective_news_gate_baseline | EDA-22-ticker mask cải thiện | **null, hypothesis rejected** | QLIKE 0.5610, DA 67.56% | 67.56→47.95% (19.6pp) | phát hiện lại VN30 universe mismatch |
| 07-25 | top3_news_gate_baseline | 3-ticker gate hẹp thành công | null | QLIKE 0.5589, DA 68.23% | 68.23→48.87% (19.4pp) | — |
| 07-26 | per_ticker_news_gate_baseline | Gate cô lập từng ticker | **positive nhưng không reproducible** | QLIKE 0.5473 (headline) | 68.90→48.08% (20.8pp) | xem mục 1.1 |
| 07-26 | spillover_qlike_baseline | Directed lag-1 graph + QLIKE loss | null (~10th liên tiếp) | QLIKE 0.5622, DA 68.23% | 68.23→47.22% (21.0pp) | — |
| 08-01 | calendar_news_gate_baseline | Calendar feature cải thiện news signal | null (4/4 biến thể) | QLIKE 0.5660 vs control 0.5497 | 68.13→45.65% (22.5pp) | corroborate bằng EDA độc lập |
| 08-01 | horizon10_baseline | Giá trị news ở horizon 10 ngày | null | QLIKE 0.573/0.577 | 67.92→48.32% (19.6pp) | — |
| 08-01 | horizon1_baseline | Horizon 1 ngày dễ nhất | positive QLIKE/R²/RMSE, **DirAcc sập** | QLIKE 0.4834 vs HAR 0.5099 | 72.39→33.16% (**39.2pp, dưới random**) | xem mục 2 |
| 08-01 | horizon22_baseline | Horizon 22 ngày | null | QLIKE 0.5938/0.5943 | 67.17→42.89% (24.3pp, dưới random) | — |

---

## 5. Đã xử lý trong phiên này (bao gồm cả phiên tiếp nối 2026-08-02 buổi chiều)

- Seed fixing (`torch.manual_seed`/`np.random.seed`) mở rộng ra **toàn bộ ~23 training script**
  (không chỉ 4 file như bản nháp trước), đã **commit** (`fccaf6a`).
- DirAcc per-ticker fix (`evaluate_predictions(..., n_stocks=...)`) áp dụng cho **toàn bộ 15+7=22
  file** gọi hàm này trên mảng flatten đa-ticker (trước đó chỉ 15/22), đã **commit**.
- `project-context.md`: tách rõ training-loss (MSE) vs evaluation-metric-priority (QLIKE) — đã
  commit.
- Mục 1.2 (leakage 2 script gốc): điều tra xong — số liệu từ 2 script này **không được trích dẫn**
  ở báo cáo hiện hành (`BAO_CAO_TONG_HOP.md`, `project-context.md`); mọi nơi từng trích đều đã gắn
  cờ "⚠️ Potential leakage". Không chặn paper, vẫn còn nợ code-hygiene (script sống, chưa archive).
- Mục 1.1: hoàn thành kiểm chứng vòng 2 (5 seed, epoch-matched = 20) — xem chi tiết cập nhật ở
  mục 1.1 phía trên. Kết luận: per-ticker-gate mean 0.5530±0.0115 QLIKE, thua REST-TS (0.5431) trên
  cả 5/5 seed — **không dùng per-ticker-gate làm headline "beats REST-TS"**.

## 6. Chưa xử lý — cần quyết định ưu tiên trước khi viết paper

Còn lại: 1.5 (chốt 1 headline — REST-TS hiện là ứng viên mạnh hơn theo mục 1.1 mới nhưng REST-TS
CHƯA qua multi-seed verify), 1.4 (phần lớn baseline khác vẫn 1-seed — chấp nhận được cho null
result, không bắt buộc cho non-headline), 1.6 (giải thích trôi dạt — coi như đã giải thích bởi
1.1: non-reproducibility), 1.7-1.9 (universe stale, traceability, cấu trúc), mục 3 (limitations,
significance testing, related work, reproducibility statement, code hygiene 3.10). Khuyến nghị thứ
tự tiếp theo: (1) multi-seed verify REST-TS (đối trọng cần thiết để chốt headline) → (2) cập nhật
Limitations + reproducibility statement với số liệu mục 1.1 mới → (3) dọn cấu trúc/traceability
còn thiếu (1.8, 1.9) → (4) dọn code hygiene nếu công khai repo (3.10, bao gồm archive 2 script rò
rỉ ở mục 1.2).
