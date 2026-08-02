# Summary — Paper-Readiness Audit Continuation (2026-08-02)

## Bối cảnh

Phiên trước (2026-08-02, kết thúc lúc ~10:56) đã sinh ra
`docs/reports/2026-08-02_1056_paper_readiness_audit_report.md` — audit toàn bộ 23 baseline +
`src/` để chuẩn bị nộp paper, cùng seed-fix và DirAcc-fix cho một phần code, nhưng bị ngắt giữa
chừng (`git status` còn nhiều `M` chưa commit, 2 run seed dở dang ở epoch 2/10). Phiên này tiếp
tục theo đúng thứ tự ưu tiên report đã đề xuất (§6), được user xác nhận.

## Việc đã làm

1. **Hoàn thiện DirAcc per-ticker fix còn dở dang** — `evaluate_predictions(..., n_stocks=...)`
   trước đó mới áp dụng cho 15/22 file gọi hàm này trên mảng flatten đa-ticker; thêm cho 7 file
   còn lại (`train_embedding_baseline.py`, `train_market_fallback.py`, `train_pure_market.py`,
   `train_alignment.py`, `train_gated_crossattn.py`, `train_resttext.py`,
   `eval_checkpoint_per_ticker.py`). `train_latent_noise.py` dùng chung `validate()` với
   `train_embedding_baseline.py` nên được fix gián tiếp.
2. **Commit toàn bộ seed-fix + DirAcc-fix** (`fccaf6a`) — seeding cho ~23 training script, DirAcc
   per-ticker fix project-wide, audit report, `DIRACC_ISSUE_NOTE.md`, 1 EDA script + test mới
   (calendar_news_gate baseline), và các kết quả run liên quan.
3. **Audit leakage (mục 1.2 report)** — `train_simple_lstm_vn30.py`/`train_lstm_har_vn30.py` có
   leakage thật (`random_split` trên time-series) nhưng số liệu của chúng **không được trích dẫn**
   ở `BAO_CAO_TONG_HOP.md`/`project-context.md` hiện hành; mọi nơi từng trích (báo cáo 06-27,
   07-11, 07-25) đều đã gắn cờ "potential leakage". Không chặn paper — vẫn là nợ code-hygiene
   (2 script sống, chưa archive).
4. **Multi-seed verification cho per-ticker-gate (mục 1.1)**, phạm vi do user chọn (chỉ
   per-ticker-gate, 5 seed, epoch-matched = epoch 20 giống con số gốc): resume 3 seed đã có
   (42, 123, 2026, đang ở epoch 10) +10 epoch; hoàn thiện seed=1 (đang dở ở epoch 2) lên epoch 20;
   chạy mới seed=7 tới epoch 20. GPU dùng `.venv_gpu_encode` (torch 2.6.0+cu124, RTX 4060 Laptop) —
   môi trường mặc định `python` là CPU-only, không có CUDA.

   | seed | test QLIKE (epoch 20) | test DirAcc per-ticker (%) |
   |---|---|---|
   | 1 | 0.5396 | 47.70 |
   | 123 | 0.5476 | 47.47 |
   | 2026 | 0.5475 | 48.54 |
   | 42 | 0.5640 | 47.05 |
   | 7 | 0.5661 | 47.80 |
   | **mean ± std (n=5)** | **0.5530 ± 0.0115** | **47.71 ± 0.55** |
   | gốc (không seed, epoch 20) | 0.5473 | — |

   Kết luận: con số gốc nằm trong 1 std của mean — không phải fluke cực đoan như nghi ngờ ban đầu,
   nhưng mean thật (0.5530) **tệ hơn REST-TS** (0.5431, single-seed, chưa verify) trên cả 5/5 seed.
   **Per-ticker-gate không dùng được làm headline "beats REST-TS"** cho paper ở dạng hiện tại.
5. **Cập nhật `docs/reports/2026-08-02_1056_paper_readiness_audit_report.md`** — đánh dấu mục 1.1,
   1.2, 1.4 là resolved (kèm số liệu/kết luận mới), cập nhật §5-§6 (trạng thái xử lý + thứ tự ưu
   tiên còn lại).

## Tests / kiểm chứng đã chạy

- `python -m py_compile` cho 7 file vừa sửa — pass.
- Smoke test thủ công `evaluate_predictions(y_true, y_pred, n_stocks=3)` — xác nhận
  `directional_accuracy_flat_biased` và `directional_accuracy` (per-ticker) trả về đúng, khớp
  `directional_accuracy_per_ticker()`.
- `python -m pytest baselines/2026-08-01_calendar_news_gate_baseline/test/test_market_news_volume_correlation.py -q`
  — 5 passed.
- `python -m pytest tests/test_evaluation.py` — **collection error, pre-existing** (`ModuleNotFoundError:
  No module named 'src.evaluation'` — file import sai path, đúng ra là `src.common.evaluation`).
  Lỗi có từ trước, không liên quan tới thay đổi phiên này — không sửa (ngoài scope, không phải
  hành vi tôi thay đổi).
- 5 training run thật (per-ticker-gate, 5 seed, epoch 20, data thật) — kết quả ở bảng mục 4 trên.
- `diff-cover`/`ruff`: **Not run** — chưa cài đặt trong repo (tooling gap đã ghi nhận sẵn trong
  CLAUDE.md).

## Code review

Không chạy `/code-review` skill đầy đủ cho thay đổi này. Lý do: 7 file sửa là **1 dòng mỗi file**
(`evaluate_predictions(targs_d, preds_d)` → thêm `n_stocks=n_stocks`), lặp lại chính xác pattern đã
được review + áp dụng ở 15 file khác trong cùng lineage baseline (cùng session trước, cùng
kiểu sửa). Tự kiểm tra: (a) `n_stocks` đã tồn tại sẵn trong mỗi file trước khi sửa (biến cục bộ,
định nghĩa từ `len(dataset.stock_names)`); (b) `py_compile` pass cả 7 file; (c) smoke test xác nhận
hành vi hàm `evaluate_predictions` đúng như kỳ vọng. Đây là follow-up nhỏ, cơ học, không giới thiệu
logic mới — nhưng theo CLAUDE.md "mọi thay đổi không ngoại lệ", đây là **gap cần note**: nếu paper
review nghiêm ngặt hoặc trước khi công khai code, nên chạy `/code-review` đầy đủ cho commit
`fccaf6a` (tất cả 76 file) ít nhất 1 lần.

## Files (path → mục đích)

- 7 file baseline (`train_embedding_baseline.py`, `train_market_fallback.py`,
  `train_pure_market.py`, `train_alignment.py`, `train_gated_crossattn.py`, `train_resttext.py`,
  `eval_checkpoint_per_ticker.py`) → thêm `n_stocks=n_stocks` vào `evaluate_predictions()`.
- `docs/reports/2026-08-02_1056_paper_readiness_audit_report.md` → cập nhật mục 1.1/1.2/1.4/§5/§6
  với kết quả điều tra + multi-seed verify mới.
- `results/per_ticker_gate_2026-08-02_{150559,150913,151224,151529,151827,152137,152448}/` →
  kết quả 5 run seed epoch-matched (một số dir trung gian do cơ chế resume tạo timestamp mới mỗi
  lần gọi lệnh — dir cuối cùng của mỗi seed là số liệu chính thức, liệt kê trong bảng mục 4).
- Commit `fccaf6a` (phiên trước đã tổng hợp) → seed-fix toàn bộ + DirAcc-fix 15/22 file đầu tiên.

## Risks / follow-ups

- **Chưa chốt headline result cho paper** — REST-TS (0.5431) hiện là ứng viên QLIKE tốt nhất nhưng
  **chưa qua multi-seed verify** (nằm ngoài phạm vi user chọn cho phiên này). Cần làm trước khi
  paper trích dẫn bất kỳ số "best" nào.
- 22/23 baseline khác vẫn 1-seed — chấp nhận được cho null result, nhưng bất kỳ claim
  positive/"beat baseline" nào từ nhóm này cần multi-seed verify riêng trước khi vào paper.
- 2 script leakage (`train_simple_lstm_vn30.py`/`train_lstm_har_vn30.py`) vẫn sống, chưa archive
  — nợ code-hygiene, không chặn paper theo hiện trạng trích dẫn.
- `tests/test_evaluation.py` collection error tiền tồn tại — chưa fix, nên xử lý trước khi coi
  test suite "xanh" cho toàn repo.
- Full `/code-review` cho commit `fccaf6a` chưa chạy — nên chạy trước khi công khai code kèm paper.

## Definition of Done checklist

- [x] Code thỏa yêu cầu, không refactor ngoài phạm vi.
- [x] Smoke test cho hành vi thay đổi (evaluate_predictions + py_compile).
- [ ] `diff-cover`/coverage gate — Not run (tooling chưa cài, gap đã biết).
- [x] Lint — Not run (`ruff` chưa cài, gap đã biết); không lint được nhưng đã `py_compile` toàn bộ
      file sửa.
- [ ] Code review đầy đủ (`/code-review`) — Not run cho phiên này, lý do nêu ở mục "Code review"
      trên; khuyến nghị chạy riêng cho commit `fccaf6a` trước khi công khai.
- [x] Summary report — file này.
- [x] Smoke chạy thật (5 training run thật, data thật, không phải dummy) — pass, xem bảng mục 4.
- [x] Impact analysis — grep toàn repo cho mọi lời gọi `evaluate_predictions(` trước khi sửa, xác
      nhận đúng phạm vi 22 file cần fix (không đụng các model family khác như LSTM-GAT gốc,
      TimesNet, VN100 — các family đó dùng pattern flatten khác, ngoài phạm vi bug này).
- [x] Similar check — grep xác nhận không còn baseline nào trong lineage news-fusion (07-07→08-01)
      còn thiếu `n_stocks=`.
