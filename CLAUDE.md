Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Spec-Driven Development (SDD)

> Áp dụng theo yêu cầu user 2026-07-15. Nguồn: github/spec-kit `spec-driven.md`
> (https://github.com/github/spec-kit/blob/main/spec-driven.md), bài tóm tắt tiếng Việt trên Viblo,
> và Microsoft "Spec-Driven Development for AI-Native Engineering"
> (https://developer.microsoft.com/blog/spec-driven-development-ai-native-engineering).

**Nguyên tắc cốt lõi:** Spec là nguồn chân lý (source of truth), KHÔNG phải code — code chỉ là bản
dịch của spec sang 1 ngôn ngữ/framework cụ thể. "Spec quality = output quality." Spec phải đủ chính
xác, đầy đủ, không mơ hồ để sinh ra hệ thống chạy được — chỗ chưa rõ PHẢI đánh dấu
`[NEEDS CLARIFICATION]` và hỏi user, KHÔNG tự giả định rồi code (siết §1 Think Before Coding thành
artifact văn bản bắt buộc, không chỉ nói miệng).

**Vòng đời bắt buộc cho mọi feature/baseline mới (mapping vào cấu trúc §3.F đã có — KHÔNG tạo
folder/artifact trùng lặp):**

1. **Constitution** — Đã có sẵn: chính file `CLAUDE.md` này là constitution của project (nguyên tắc,
   tiêu chuẩn, gate). Spec/plan mới phải tuân theo rule đã có ở đây; conflict thì CLAUDE.md thắng —
   sửa CLAUDE.md trước rồi mới sửa spec.
2. **Specify** — Viết spec TRƯỚC khi code: mục tiêu, input/output, acceptance criteria, edge case,
   go/no-go. Dùng `requirements/requirements.md` (đã bắt buộc §3.F) làm spec.md.
3. **Clarify** — Rà lại spec tìm chỗ mơ hồ/thiếu, đánh dấu `[NEEDS CLARIFICATION]`, hỏi user resolve
   trước khi sang Plan — không suy diễn thay user.
4. **Plan** — Kiến trúc, data flow, quyết định design, dependency. Dùng `design/design.md`
   (đã bắt buộc §3.F) làm plan.md. Qua 3 gate trước khi implement:
   - **Simplicity Gate:** không thêm project/abstraction/config vượt mức cần (per §2 Simplicity First).
   - **Anti-Abstraction Gate:** dùng thẳng thư viện/framework có sẵn, không tự wrap khi không cần.
   - **Performance/Batching Gate:** với code train/inference/data-processing, design PHẢI nêu rõ
     chiến lược batch + song song (per §Performance & batching ENFORCED bên dưới) — KHÔNG mặc định
     vòng lặp per-item batch=1 trên main thread. Không có kế hoạch batch → không qua gate.
   - Nếu buộc phải phá 1 trong 3 gate → ghi rõ lý do (complexity tracking) trong `design.md`.
5. **Tasks** — Tách plan thành danh sách task đánh số, mỗi task có tiêu chí verify được (test/smoke
   pass) — LIỆT KÊ rõ trước khi code (siết §4 Goal-Driven Execution thành checklist tường minh).
6. **Implement** — Test-First bắt buộc: viết test → xác nhận test FAIL → implement code → test pass
   → refactor nếu cần. KHÔNG viết implementation code trước khi có test tương ứng cho hành vi đó.
7. **Validate** — Xác nhận output khớp spec (acceptance criteria ở bước Specify) trước khi coi done —
   trùng với Definition of Done ở dưới, không làm lại 2 lần.

**Right-sizing (KHÔNG áp dụng máy móc mọi việc):** Full vòng đời 7 pha dùng cho baseline mới (§3.F)
và thay đổi kiến trúc lớn (vd LSTM-GAT, TimesFM, tầng model mới). Fix nhỏ 1-dòng / sửa doc-typo /
lint fix thì dùng judgment (per tradeoff đầu file) — không bắt viết đủ 7 pha cho việc trivial.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project Quality Rules

> Project-agnostic quality gate. Áp dụng cho **MỌI thay đổi** (code, docs, config, scripts) — không ngoại lệ. (Mở rộng §3.F — baseline-specific rules vẫn giữ, đây là chuẩn chung cho mọi change.)

## Definition of Done
Task chỉ "done" khi TẤT CẢ đúng:
- **Code** thỏa mãn đúng request; không refactor không liên quan (per §3 Surgical).
- **Tests + coverage:** khi đổi behavior, viết/chạy unit test và đạt coverage gate — chi tiết + lệnh ở section **Testing quality rules (ENFORCED)** ngay dưới đây (C0=100% / C1≥80% trên CHANGED lines, đo bằng **diff-coverage**, KHÔNG phải total coverage). Change phải staged/committed để diff đo được.
- **Checks run:** bắt buộc chạy test + lint. KHÔNG claim "pass" nếu chưa chạy thật.
- **Lint scope:** exclude vendored/generated/third-party (`.agents .claude _bmad archive data`).
- **Audit/Review scope:** `archive/` (any depth) is retired code/data, intentionally out of scope
  for ALL audits, code review, adversarial review, and any AI-driven "find issues" pass — not
  just lint. Do not report findings, propose fixes, or flag bugs for anything under `archive/`
  unless the user explicitly asked to review `archive/` itself. When briefing a subagent for a
  repo-wide audit, state this exclusion explicitly in the prompt — a subagent given only a
  directory to explore will not otherwise know `archive/` is off-limits. `archive/README.md`
  carries the same notice for anyone who lands there directly.
- **Code review (LUÔN, mọi change):** chạy `/code-review` (hoặc adversarial PR review) + xử lý findings trước khi done. **Bắt buộc mọi thay đổi — kể cả docs/config/scripts — không ngoại lệ.** Tóm tắt result + actions trong summary report.
- **Performance/batching (khi đụng train/inference/data-processing):** phải đạt §Performance & batching (ENFORCED) — batch tối đa (không batch=1 mặc định), ưu tiên GPU, song song hoá, tận dụng RAM/VRAM; code review flag batch=1/GPU-underuse/main-thread-only là finding phải xử lý. Ghi kết luận hiệu năng vào summary report.
- **Summary report:** sinh `docs/reports/<YYYY-MM-DD_HHMM>_summaryOfUpdate_report.md` (context-appropriate, không rigid template).
- **Push remote ngay sau mỗi task done:** khi 1 task đã verify xong (có evidence — test pass,
  smoke pass, hoặc kết quả đo được cụ thể), commit VÀ `git push origin master` ngay, không đợi
  user nhắc, không gộp nhiều task rồi push 1 lần cuối session. Áp dụng cho AI/session bất kỳ làm
  việc trên repo này — không chỉ push khi được hỏi (khác Git Safety Protocol mặc định, override
  có chủ đích theo yêu cầu user 2026-08-02). Nếu push thất bại (remote đã đổi khác) — dừng lại,
  hỏi user, KHÔNG force-push.
- **Smoke (gate):** ≥1 smoke test (tag `smoke`) boot pipeline/app + 1 happy-path. **Phải pass trước done.** Nếu cần infra ngoài cũng phải chạy.
- **Data-quality gate (Pandera + Evidently) — BẮT BUỘC trong quality gate:** quality gate PHẢI chạy 2 tool này và capture evidence vào summary report + task dashboard, KHÔNG được bỏ qua/skip im lặng:
  - **Pandera** — schema validation trên `data/processed` (OHLCV/HAR non-neg, high≥low, date monotonic) + news panel (`check_schema()` trong `scripts/quality_gate/`).
  - **Evidently** — drift report train-vs-test trên feature (`check_drift()`), sinh `drift.html` artifact.
  - Lệnh: `python scripts/quality_gate/run_quality_gate.py` (chạy cả LINT/TESTS/SCHEMA/DRIFT), hoặc gọi `check_schema()`/`check_drift()` trực tiếp cho phần data-quality.
  - **Scope bắt buộc:** MỌI change đụng data/features/manifest/pipeline-train (vd baseline mới, đổi dataset, đổi manifest). Change thuần code KHÔNG đụng data: ghi `N/A (no data change)` kèm lý do trong report — không claim đã chạy khi chưa chạy. Cả 2 tool đã cài (pandera 0.32.1 + evidently 0.7.21, verified trên pandas 3.0).
- **Raw-data ingestion quality tests — BẮT BUỘC + ENFORCED bằng harness (áp dụng theo yêu cầu user 2026-08-16):** mỗi khi **crawl/load/append dữ liệu raw mới** (vd `data/raw/prices`, `data/raw/prices/vn100`, hay nguồn mới), TRƯỚC khi commit/dùng PHẢI: (1) chạy `python -m pytest tests/test_raw_prices_quality.py tests/test_processed_data_quality.py` và **pass hết** (schema, ngày monotonic/weekday-only, OHLC hợp lệ high≥low + open/close trong [low,high] + giá dương, không NaN/inf, phát hiện backfill), (2) reprocess (`python -m src.common.process_parkinson_pipeline`) + đồng bộ sang bản data mà code đọc (worktree nếu train ở worktree), (3) chạy data-quality gate (Pandera+Evidently). **Enforce tự động:** pre-push hook (`scripts/git_hooks/pre-push` step 4) tự chạy 2 test này và **block push** khi data bị đụng mà test fail — KHÔNG được `QG_SKIP` để lách trừ khi có lý do chính đáng ghi trong report. Fix data lỗi cho đúng (vd nonpositive OHLC → dùng max/min trên giá dương; backfill trước niêm yết → cắt về ngày niêm yết thật), KHÔNG bỏ qua im lặng.
- **Impact analysis:** trước change non-trivial, xác định blast radius — grep callers/dependents, check registration/integration points, note cross-repo consumers. Tóm tắt affected + verified. Flag risk nếu blast radius lớn mà chưa test đủ.
- **Similar check:** sau fix/pattern change, grep cùng idiom/duplicate trong repo + sibling repos. Apply cùng change nơi hợp lệ, hoặc list remaining as follow-up. Đừng fix 1/N copy silent.

## Testing quality rules (ENFORCED)

> Nguồn: mượn từ project sibling `thesis/data_eda` (CLAUDE.md), áp dụng theo yêu cầu user 2026-07-15.

Coverage % đơn thuần KHÔNG chứng minh chất lượng — adversarial review có thể tìm bug thật (vd: date mass-NaT, tz-aware crash, NaN bị đếm nhầm thành duplicate, dead code, thiếu acceptance criteria) ngay cả khi line coverage = 100%. Do đó, với MỌI thay đổi behavior:

- **Coverage gate (BẮT BUỘC), tách C0/C1:**
  - **C0 (line coverage) = 100%** trên CHANGED lines (mọi dòng thực thi phải được test chạy qua).
  - **C1 (branch coverage) ≥ 80%** trên CHANGED lines (mọi nhánh điều kiện phải được cover; >80% chấp nhận khi cover toàn bộ bất khả thi — phải nêu lý do).
  - Lệnh: `python -m pytest --cov=src --cov-report=xml -q && diff-cover coverage.xml --fail-under=100` (gate C0); sau đó soi report coverage.xml/html để xác nhận C1 ≥80% trên các dòng đổi.
  - `pytest --cov` xanh KHÔNG đủ — `diff-cover` trên changed lines mới là gate thật.
- **Test I/O runner, không chỉ pure helper.** Unit test hàm thuần dễ bỏ sót bug ở code load-data/ghi-report/train-loop. Mọi hàm `run_*()`/report-builder/`train_epoch`-style phải có ít nhất 1 test tích hợp (monkeypatch path hoặc dùng tmp fixture) trước khi coi task done.
- **Data-pipeline test phải có real-data-sample smoke.** Synthetic fixture bỏ sót lỗi encoding (UTF-8 vs cp1252), mixed date format (ISO vs DD/MM), mixed timezone, schema drift giữa nguồn crawl khác nhau (đúng loại lỗi từng gặp ở `aggregate_news_sources.py`). Ít nhất 1 test/phase đọc 1 lát cắt nhỏ dữ liệu thật (không phải toàn bộ) và assert chạy không exception + output hợp lý.
- **Code review = 3-layer (Blind Hunter + Edge Case Hunter + Acceptance Auditor) qua `/code-review`, chạy TRƯỚC khi coi done.** Self-review KHÔNG thay thế được. Xử lý hết finding critical/major trước khi done; ghi minor thành follow-up trong summary report. **Với code train/inference/data-processing: review PHẢI có 1 lăng kính hiệu năng** — bắt batch=1 anti-pattern, GPU under-utilization, transfer/sync mỗi step, loop chỉ main-thread (per §Performance & batching ENFORCED).

**Tooling status (cập nhật 2026-08-29):** `diff-cover` ĐÃ cài (10.5.0) + pytest-cov (7.1.0) và ĐÃ wired vào
pre-push gate (step 2): mỗi push sinh `coverage.xml` (chạy test cạnh file đổi dưới GPU venv, `--cov-branch`)
rồi chạy 2 gate — **C0 line `diff-cover --fail-under=100`** và **C1 branch `--branch-coverage --fail-under=95`**
trên changed lines so với `@{upstream}`. Env knob `QG_MIN_COVER` / `QG_MIN_BRANCH`. Dòng entry-driver
`main()` không unit-test được đánh `# pragma: no cover`; residual chỉ là các nhánh đó.

## Summary report (per change)
Khi done, sinh markdown summary ngắn, context-appropriate → `docs/reports/<YYYY-MM-DD_HHMM>_summaryOfUpdate_report.md`.
- Fit THIS change: bỏ phần không relevant — **trừ code review, luôn required + tóm tắt.**
- Cover (as applicable): what changed, files (path → purpose), tests + coverage %, code-review result + actions, commands run thật, risks/follow-ups, DoD checklist.
- Chỉ ghi gì thật xảy ra; `Not run` (+lý do) cho cái skip.

## Văn phong báo cáo — khách quan (ENFORCED, áp dụng MỌI report/tài liệu, không riêng summary report)

> Áp dụng theo yêu cầu user 2026-08-01. Lý do vi phạm trước đó: file/nội dung report từng dùng
> cách xưng hô cá nhân ("thầy") và ngôn từ tự khẳng định ("báo cáo trung thực") — không phù hợp
> văn phong tài liệu kỹ thuật khách quan.

- **KHÔNG xưng hô/gọi tên vai trò cụ thể** trong nội dung hay tên file: không dùng "thầy",
  "giảng viên", "sinh viên", hay bất kỳ cách xưng hô cá nhân nào khác. Viết như tài liệu kỹ thuật
  trung lập, không phải thư/lời nhắn gửi một người cụ thể.
- **KHÔNG dùng ngôn từ chủ quan/hoa mỹ** (cảm thán, nhấn mạnh cảm xúc, tự khen). Chỉ nêu sự kiện,
  số liệu, kết luận — để dữ liệu tự nói, không cần tính từ tô điểm.
- **KHÔNG tự khẳng định "trung thực"/"thành thật"/"honest"** ở đầu hay cuối báo cáo (vd "Báo cáo
  trung thực:", "Lưu ý trung thực:"). Việc báo cáo đúng sự thật là mặc định bắt buộc (đã có rule
  "chỉ ghi gì thật xảy ra" ở trên) — không cần tuyên bố lại, tuyên bố này chính nó mang tính chủ
  quan/phòng vệ, không phải văn phong khách quan.
- **Cách viết đúng:** nêu thẳng dữ kiện + số liệu + nguồn trích dẫn (file/`results.json`), không
  thêm câu dẫn nhập mang tính cá nhân hoá hay tự đánh giá về độ tin cậy của chính báo cáo.
- Áp dụng cho: tên file, tiêu đề, mọi section trong `docs/reports/`, `docs/report_*/`,
  `code_review/`, và bất kỳ markdown nào sinh ra làm báo cáo.

## Code hygiene (mọi ngôn ngữ)
- No hidden global state / unbounded in-process caches (dùng bounded TTL/size cache; externalize shared state).
- No secrets in code (secrets manager / env).
- No hardcoded absolute local paths.
- No production logic chỉ sống trong notebook.
- **No silent degradation (ENFORCED, per root-cause 2026-08-16):** data/feature code KHÔNG được âm
  thầm trả giá trị "trung tính" (zeros/NaN→0/fallback) khi input thiếu/sai config — phải **fail loud**
  (raise) hoặc bounded-allowlist (chỉ dung thứ ≤ ngưỡng nhỏ đã ghi rõ, vượt là raise). Lý do: một
  wrapper set `_PROCESSED`=VN100 nhưng quên `_PRICE_DIR` khiến `volume_zscore` bị zero âm thầm cho
  71/104 mã mà không gate nào bắt được (silent semantic bug). Guard: `augment_split_frames`
  `_check_price_coverage` raise khi >1 mã thiếu raw.
- **Coupled config phải validate cùng nhau / dùng CLI có test, KHÔNG monkeypatch global rời rạc:** các
  global đi cặp (`_PROCESSED`↔`_PRICE_DIR`, SEQ, HORIZON) khi đổi universe phải set đồng bộ + có assert
  nhất quán; ưu tiên `--processed/--price-dir/--universe` (đã test) hơn là patch global trong wrapper
  throwaway (đường thoát khỏi quality gate).
- **Experiment/wrapper scripts cũng phải qua chất lượng:** script thí nghiệm (dù throwaway) đụng
  train/eval phải có smoke-assert bất biến cơ bản (vd feature không bị all-zero cho mã hợp lệ) trước
  khi tin số; và **test của baseline giao nộp chạy trong pre-push gate** (hook step 5, GPU venv) — không
  để test baseline nằm ngoài gate.

## Performance & batching (ENFORCED — quality gate + code review)

> Áp dụng theo yêu cầu user 2026-08-16. Lý do: đo thực tế cho thấy train **batch=1/snapshot** làm
> GPU (RTX 4060) **~97% nhàn rỗi** (batch=1 và batch=32 cùng ~10ms/step) — nghẽn ở overhead
> Python/kernel-launch/host→device copy, KHÔNG phải compute. Batch lại cho **~20–32× miễn phí** trên
> GPU hiện tại (xem `docs/reports/2026-08-16_training_acceleration_eval.md` +
> `2026-08-16_perf_audit_optimization_plan.md`). Vì vậy hiệu năng là 1 tiêu chí done, không phải
> tối ưu-sau.

Mọi code **train/inference/data-processing** (và design/plan sinh ra nó) PHẢI, theo thứ tự ưu tiên:
1. **Batch tối đa, KHÔNG batch=1 mặc định.** Xử lý nhiều mẫu/snapshot mỗi bước (batched tensor
   `[B, ...]`, batched/block-diagonal adjacency + mask-aware loss cho graph). Vòng lặp Python per-item
   trên main thread là anti-pattern — chỉ chấp nhận khi có lý do ghi rõ (vd phụ thuộc tuần tự thật).
2. **Ưu tiên GPU rồi mới CPU.** Giữ tensor trên GPU, giảm tối đa transfer host↔device và điểm đồng bộ
   (`.item()`/`.cpu()` trong vòng nóng); dùng `pin_memory`+`non_blocking`, cân nhắc AMP.
3. **Song song hoá, không chỉ main thread.** Dùng `DataLoader(num_workers>0)`/multiprocessing/thread
   cho load-data + tiền xử lý overlap với compute; tách việc độc lập ra process/thread.
4. **Tận dụng RAM + VRAM sẵn có** (vd 32GB RAM): preload dữ liệu lên device/pinned buffer, batch lớn
   trong giới hạn VRAM.

**Enforce:**
- **Quality gate + code review:** mỗi change đụng train/inference/data-processing phải được rà điểm
  hiệu năng này; `/code-review` PHẢI flag batch=1 anti-pattern / GPU under-utilization / per-step
  transfer / main-thread-only loop như finding (mặc định MAJOR nếu ở vòng nóng). Không xử lý →
  không done. Ghi kết luận hiệu năng (đã batch/song song ra sao, hoặc lý do không) vào summary report.
- **Design/Plan:** qua **Performance/Batching Gate** ở §5 SDD trước khi implement.
- **Không** hy sinh tính đúng/không-leakage để lấy tốc độ; batch phải giữ nguyên ngữ nghĩa (mask,
  thứ tự thời gian, per-ticker scaler).

## Training policy (experimentation) ⭐
Khi **thử nghiệm** model (không phải final run):
- **Default 5-10 epoch** — KHÔNG train >10 epoch khi chưa có quyết định.
- **Báo cáo mỗi 5 epoch:** in val metrics (DirAcc, RMSE, R², QLIKE) + **learning curve image** (§3.C) sau mỗi 5 epoch để xem tiến độ quyết định train tiếp hay dừng.
- **>10 epoch cần đồng ý rõ ràng** của user (dựa trên kết quả 5/10 epoch).
- **Full run (vd 40/70 epoch)** chỉ khi user approve sau khi xem 5/10 epoch.
- Lý do: training dài tốn thời gian; nhiều baseline no-lift (4/4 news + body) → checkpoint sớm đỡ lãng phí.

**Over/under-fit evidence BẮT BUỘC + ENFORCED (user mandate 2026-08-29):** mỗi lần train phải **capture
evidence** để CHỨNG MINH (không chỉ khẳng định) model không overfit/underfit, và evidence này bị gate chặn:
- **Capture trong lúc train:** result.json PHẢI có `train_metrics` + `val_metrics` (cạnh `metrics`=test) cho
  từng model + `fit_diagnostics` (verdict per model) + `learning_curves` (train/val MSE mỗi epoch, per seed).
  `run_masked_rich.train_masked_rich(return_splits=True)` + `run()` đã sinh sẵn; logic verdict ở
  `scripts/quality_gate/overfit_check.py::classify_fit` (overfit = val→test QLIKE xấu đi >25% hoặc train→test
  R² drop >0.20; underfit = train_r2 & test_r2 đều < floor).
- **Gate chặn (pre-push):** `scripts/quality_gate/check_overfit_evidence.py` chạy trên mọi `result.json`
  masked_rich trong push diff — thiếu evidence hoặc model overfit/underfit → **BLOCK**. Test:
  `test_overfit_check.py`, `test_check_overfit_evidence.py` (chạy mỗi push).
- Các delivered result.json cũ (test-only, trước mandate) không nằm trong push diff nên không bị retro-check;
  nhưng mọi lần train/commit result.json MỚI từ nay phải kèm evidence.

## Ablation study — LEAVE-ONE-OUT (ENFORCED)

> Áp dụng theo yêu cầu user 2026-08-15. Lý do: một pass trước làm ablation kiểu incremental
> (cộng dần HAR→+LSTM→+news→+gate→+graph) trong khi user muốn ablation ĐÚNG nghĩa = gỡ từng
> thành phần khỏi model đầy đủ. Rule này để không tái lập.

- **Ablation study BẮT BUỘC theo kiểu LEAVE-ONE-OUT:** build **model ĐẦY ĐỦ (full)** trước, rồi
  tạo các biến thể bằng cách **GỠ ĐÚNG 1 thành phần** khỏi full (Full, Full−graph, Full−gate,
  Full−news, …). Mỗi biến thể = full thiếu 1 cái → đo *"thiếu thành phần đó thì model tệ đi bao
  nhiêu"* (đóng góp biên của thành phần KHI CÓ MẶT tất cả cái khác).
- **KHÔNG dùng incremental (cộng dần từ nhỏ) LÀM ABLATION CHÍNH.** Incremental ladder
  (baseline → +A → +B → …) là *component-addition / ladder study*, trả lời câu hỏi khác và cho số
  khác với leave-one-out ở model phi tuyến (do tương tác giữa thành phần). Có thể báo cáo kèm như
  phụ, nhưng **ablation study mặc định = leave-one-out**.
- **Cách đo đóng góp:** `effect(X) = metric(Full) − metric(Full−X)` trên cùng test set, cùng basis,
  cùng positivity floor; báo cáo dấu rõ ràng (giảm loss = thành phần giúp). So sánh thống kê
  (Diebold-Mariano/paired) giữa Full và Full−X.
- **Thành phần chung nhiều nhánh (đã đính chính 2026-08-18 — model giao nộp `baselines/2026-08-15_volatility/code/model.py`):**
  5 node feature `[parkinson_volatility, har_weekly, har_monthly, market_pk, volume_zscore_20]` dùng
  CHUNG cho cả nhánh LSTM và nhánh GAT, nhưng kiến trúc là **PARALLEL (Track-A), KHÔNG nối tiếp**:
  - **Nhánh LSTM** đọc chuỗi SEQ ngày của 5 feature (`price_lstm`, `model.py:75`) → `h_lstm` [B,N,64].
  - **Nhánh GAT** đọc **feature THÔ tại ngày cuối t** (`node_raw = price[:,:,-1,:]`, `model.py:89`;
    `GATLayer(price_dim=5,...)`, `model.py:37`) — **KHÔNG dùng output của LSTM** (comment gốc
    `model.py:29-32`). Vì vậy **SEQ (lookback) KHÔNG ảnh hưởng nhánh GAT** — graph luôn thấy 1 vector
    ngày t bất kể SEQ.
  - Head ghép song song `cat([h_lstm(64), h_gnn(256), gated_news(64)])` (`model.py:92`).
  - (Note cũ ghi "GAT nhận node representation từ LSTM" là SAI với model này — mô tả biến thể khác/cũ.)
  - **HAR features tính per-day, độc lập SEQ:** `har_weekly`=rolling mean 5 ngày, `har_monthly`=22 ngày,
    là cột tính sẵn; SEQ chỉ cắt cửa sổ `feature_values[start:start+SEQ]` (`data.py:437`), nên har_monthly
    vẫn hiện diện mỗi timestep kể cả khi SEQ=5.
  - edge (graph) là quan hệ giữa node (vol→PK Top-5), KHÔNG phải node feature. Khi mô tả ablation phải
    phân biệt rõ NODE feature vs EDGE.

## Per-project setup (stack specifics — chỗ DUY NHẤT hardcoded stack)
- **Language/toolchain:** Python 3.11 (pip)
- **Test command:** `python -m pytest`
- **Coverage + diff-coverage:** pytest-cov (7.1.0) + diff-cover (10.5.0) ĐÃ cài + wired vào pre-push gate. Gate: `--cov-branch` → `diff-cover coverage.xml --fail-under=100` (C0 line) và `diff-cover ... --branch-coverage --fail-under=95` (C1 branch) trên changed lines. Lesson-regression: `tests/test_lessons_regression.py` chạy mỗi push.
- **Lint command:** `ruff check .` (ruff 0.16.1 đã cài). **Policy (review 2026-08-29):** pre-push gate
  **FAIL-CLOSED trên ruff pyflakes `F` codes** (F401 unused import, F811 redefinition, F821 undefined name —
  bug thật). Các `E`/`W` là **house-style WARN-only** cố ý: repo dùng `;`-joined statements (E702), dòng dài
  (E501), lambda (E731) cho code số gọn — không block, khớp §3 Surgical (match existing style). Field `ruff`
  trong gate JSON là full-lint status (informational); block thực sự là `ruff check --select F`.
- **Lint excludes:** `.agents .claude _bmad archive data`
- **Smoke command:** `python -m pytest -m smoke` *(cần register marker `smoke` trong pytest.ini)*
- **Code-review tool:** `/code-review` skill (Claude Code)
- **Data-quality gate (ENFORCED, xem §Definition of Done):** `python scripts/quality_gate/run_quality_gate.py` — Pandera schema (`data/processed` + news panel) + Evidently drift (train-vs-test → `drift.html`). *(pandera 0.32.1 + evidently 0.7.21 đã cài, verified trên pandas 3.0.)*
- **Python extras:** tránh bare `except` + mutable default args; dùng type hints + `pathlib`

> **Tooling status (2026-08-29):** ruff (0.16.1), pandera (0.32.1), evidently (0.7.21), pytest (9.1.1),
> **pytest-cov (7.1.0), diff-cover (10.5.0)** ĐÃ install; data-quality gate (`scripts/quality_gate/`) hoạt
> động. **Coverage gate ĐÃ wired vào pre-push (step 2): C0 line=100% + C1 branch=95% trên changed lines,
> hard-block.** Lesson-regression suite (`tests/test_lessons_regression.py`) chạy mỗi push. **Còn gap:**
> smoke marker registration cần xác nhận trong pytest.ini.

---

# Stock Volatility Prediction - VN30

**Project:** Multi-horizon volatility forecasting cho 30 VN30 stocks  
**Date:** 2026-06-19  
**Status:** Development Phase - Implementing 3-way temporal split evaluation

---

## Quick Start

```bash
# Install dependencies
pip install torch pandas numpy scikit-learn

# Process data
python process_data.py

# Train model (with proper temporal split)
python src/lstm_har_enhanced/train_with_validation.py
```

---

## 1. Project Overview

### **Objective**
Build robust volatility prediction system cho 30 VN30 stocks using daily OHLCV data, implementing HAR methodology adapted cho daily frequency.

### **Primary Target**
- **5-day ahead volatility forecast** (current focus)
- **Secondary Targets:** 1, 10, 22-day forecasts (future expansion)

### **Success Criteria**
- **RMSE:** < 0.20 cho 5-day forecasts
- **Directional Accuracy:** > 55% cho 5-day forecasts
- **Test Coverage:** 85%+ overall, 90% cho critical paths

---

## 2. ML/DS Common Rules

**This project follows ML/DS common clean code rules.**

### **External Package Reference**
**Package:** `ml-ds-common-rules`  
**Location:** `D:\bmad-projects\ml-ds-common-rules`  
**Installation:** `pip install -e D:\bmad-projects\ml-ds-common-rules`

**Quick Links:**
- 📘 [Common Rules](D:\bmad-projects\ml-ds-common-rules\COMMON_RULES.md)
- 📗 [Quick Reference](D:\bmad-projects\ml-ds-common-rules\QUICK_REFERENCE.md)
- 📕 [Integration Guide](D:\bmad-projects\ml-ds-common-rules\INTEGRATION_GUIDE.md)

### **Key Principles (from ml-ds-common-rules)**

1. **Code is read much more than written** → Write for future readers
2. **Leave code better than you found it** → Boy scouts rule
3. **Keep it simple** → Simple > Clever
4. **Match quality to maturity** → Don't over-engineer POCs

**For detailed rules:** See `ml-ds-common-rules` package documentation above.

---

## 3. Project-Specific Rules

### **Critical Rules for Volatility Forecasting**

#### **A. Temporal Data Splitting (MANDATORY)** ⭐

**CRITICAL:** Time series data MUST be split chronologically to prevent data leakage.

```python
# ✅ CORRECT - Temporal split
from src.common.temporal_split import TemporalSplitter

splitter = TemporalSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_loader, val_loader, test_loader = splitter.create_dataloaders(dataset)

# ❌ WRONG - Random split causes data leakage
train, test = torch.utils.data.random_split(dataset, [0.8, 0.2])
```

**Split ratios:**
- Train: 70% (2006-2020)
- Validation: 15% (2020-2021) - for early stopping
- Test: 15% (2021-2026) - final evaluation

**Documentation:** `docs/project/TEMPORAL_SPLIT_EVALUATION.md`

---

#### **B. Mandatory Evaluation Metrics (ALL 6)** ⭐

**CRITICAL:** Every model MUST be evaluated on ALL 6 metrics below.

```python
from src.common.evaluation import evaluate_predictions

metrics = evaluate_predictions(y_true, y_pred)
# Returns: mse, rmse, mae, r2, qlike, directional_accuracy
```

**6 Mandatory Metrics:**
1. **MSE** - Mean Squared Error (lower is better) ⭐ NEW
2. **RMSE** - Root Mean Squared Error (lower is better)
3. **MAE** - Mean Absolute Error (lower is better)
4. **R²** - Variance Explained (higher is better)
5. **QLIKE** - Academic standard cho volatility (lower is better)
6. **Dir Acc** - Directional Accuracy (higher is better)

**Output Requirements (MANDATORY):**

**1. Console Output Format:**
```python
# Validation Results
print(f"Val MSE: {val_metrics['mse']:.6f}")
print(f"Val RMSE: {val_metrics['rmse']:.6f}")
print(f"Val MAE: {val_metrics['mae']:.6f}")
print(f"Val R²: {val_metrics['r2']:.6f}")
print(f"Val QLIKE: {val_metrics['qlike']:.6f}")
print(f"Val Dir Acc: {val_metrics['directional_accuracy']:.2f}%")

# Test Results (same format)
print(f"Test MSE: {test_metrics['mse']:.6f}")
print(f"Test RMSE: {test_metrics['rmse']:.6f}")
# ... etc
```

**2. JSON Output Format:**
```python
results = {
    'validation_metrics': {
        'mse': float(val_metrics['mse']),
        'rmse': float(val_metrics['rmse']),
        'mae': float(val_metrics['mae']),
        'r2': float(val_metrics['r2']),
        'qlike': float(val_metrics['qlike']),
        'directional_accuracy': float(val_metrics['directional_accuracy'])
    },
    'test_metrics': { ... },
    'val_test_diff': {
        'mse_diff': float(mse_diff),
        'rmse_diff': float(rmse_diff),
        'mae_diff': float(mae_diff),
        'r2_diff': float(r2_diff),
        'qlike_diff': float(qlike_diff),
        'dir_acc_diff': float(dir_acc_diff)
    }
}
```

**3. Comparison Table (for validation):**
```
Metric          Validation       Test            Difference
------------------------------------------------------------
MSE             0.xxxxxx        0.xxxxxx       +0.xxxxxx
RMSE            0.xxxxxx        0.xxxxxx       +0.xxxxxx
MAE             0.xxxxxx        0.xxxxxx       +0.xxxxxx
R²              0.xxxxxx        0.xxxxxx       +0.xxxxxx
QLIKE           0.xxxxxx        0.xxxxxx       +0.xxxxxx
Dir Acc         xx.xx%          xx.xx%         +x.xx%
```

**Documentation:** `docs/project/` - See Model Evaluation Rules section

**Critical Bug Warning - Directional Accuracy:**
```python
# ❌ WRONG - Sign of values (always 100% for volatility)
dir_acc = np.mean(np.sign(y_true) == np.sign(y_pred)) * 100

# ✅ CORRECT - Sign of CHANGES
actual_changes = np.sign(np.diff(y_true))
pred_changes = np.sign(np.diff(y_pred))
dir_acc = np.mean(actual_changes == pred_changes) * 100
```

---

#### **C. Learning Curves (MANDATORY)** ⭐

**CRITICAL:** Plot learning curves cho ALL training runs to detect overfitting.

```python
# During training - PLOT EVERY 10 EPOCHS
if (epoch + 1) % 10 == 0:
    plot_learning_curves(train_losses, val_losses, save_path)
```

**Documentation:** See `ml-ds-common-rules` COMMON_RULES.md "Learning Curves and Overfitting Detection"

---

#### **D. File Organization**

**Mandatory Structure:**
```
project_root/
├── CLAUDE.md                   # This file (project rules)
├── README.md                   # Project overview
├── project-context.md          # Project background
├── src/                        # ALL code here
│   ├── common/                 # Shared utilities
│   ├── har_baseline/           # HAR baseline
│   ├── lstm_baseline/          # LSTM baseline
│   ├── lstm_har_baseline/      # LSTM-HAR
│   ├── lstm_har_enhanced/      # Enhanced LSTM-HAR
│   └── experiment/             # Experimental code
├── docs/                       # ALL docs here
│   ├── project/                 # Project docs
│   ├── lstm/                    # LSTM docs
│   └── common-rules/           # Common rules (reference)
├── results/                    # ALL results here
├── models/                     # ALL models here
└── data/                       # Data files only
```

**Generated Files Naming:**
```python
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

# Result files
results_file = f"results/enhanced_lstm_{timestamp}/"
model_file = f"models/baseline_{timestamp}/"
```

**Documentation:** See `ml-ds-common-rules` COMMON_RULES.md "File Management and Archiving"

---

#### **F. Baseline Implementation Structure (MANDATORY)** ⭐

**CRITICAL:** Mỗi lần implement 1 baseline MỚI, PHẢI tạo 1 folder riêng có timestamp, chứa đủ 4 sub-folder. KHÔNG dump code lẫn vào `src/` chung hay lẫn baseline cũ.

**Cấu trúc bắt buộc:**
```
baselines/
└── YYYY-MM-DD_<tên-baseline>/      ← timestamp = ngày bắt đầu implement
    ├── requirements/
    │   └── requirements.md          # Mục tiêu, input/output, success criteria, go/no-go
    ├── design/
    │   └── design.md                # Kiến trúc, quyết định design, data flow, file list
    ├── code/
    │   ├── __init__.py
    │   └── *.py                     # Toàn bộ code baseline (extract, dataset, model, train)
    ├── code_review/
    │   └── code_review_<YYYY-MM-DD>.md  # Kết quả review (adversarial, per CLAUDE.md mục 5)
    └── test/
        ├── __init__.py
        └── test_*.py                # Unit test + smoke test
```

**Quy tắc:**
1. **Timestamp** = ngày bắt đầu (vd `2026-07-07_embedding_baseline`). Mỗi baseline 1 folder.
2. **5 sub-folder BẮT BUỘC**: `requirements/`, `design/`, `code/`, `code_review/`, `test/`. Không bỏ sót. Code review (adversarial, theo mục 5) phải chạy TRƯỚC khi coi baseline "done".
3. **Cô lập cứng (hard isolation)**: code baseline KHÔNG sửa file/module của baseline khác hay `src/` chung. Import read-only từ `src/` thì ĐƯỢC.
4. **Code phải chạy được**: mỗi script tự bootstrap `sys.path` (project root + `code/`) vì tên folder có dấu gạch (`-`) không hợp lệ cho `python -m`. Chạy bằng `python <path>/<script>.py`.
5. **Test phải dùng `pytest`** (`pip install pytest`). Test phải chạy được bằng `pytest test/` (đặt tên hàm `test_*` để pytest auto-discover; KHÔNG dùng plain-assert runner là primary). Phải chạy với dummy data (không phụ thuộc data thật nặng) → verify kiến trúc trước khi train thật. Lệnh chuẩn: `pytest <baseline>/test/ -v`.
6. **Results/models** vẫn lưu vào `results/`, `models/` gốc (theo 3.D), KHÔNG tạo riêng trong baseline folder.
7. **Unit test issues PHẢI fix hết**: mọi test fail/error (pytest) phải được fix **hoàn toàn** trước khi baseline "done". KHÔNG skip test để qua, KHÔNG để test đỏ tồn tại. Test mới phát hiện bug → fix bug rồi verify test pass.

**Verify trước khi coi baseline "done":**
- [ ] Folder timestamp + 5 sub-folder tồn tại (requirements, design, code, code_review, test)
- [ ] `requirements.md` có success criteria + go/no-go rõ ràng
- [ ] `design.md` có data flow + quyết định design
- [ ] `code/` chạy được (smoke test pass)
- [ ] `code_review/` có kết quả review adversarial (HIGH/MEDIUM đã fix, phải fix hết)
- [ ] `test/` pass với **pytest** (`pytest test/ -v` pass; ít nhất: shape correctness + 1 property test)

---

#### **E. Overfitting Prevention (MANDATORY)** ⭐

**CRITICAL:** ALL models MUST apply anti-overfitting techniques to ensure generalization.

**3 Groups of Techniques (in priority order):**

**1. Data-Centric (Priority 1):**
- Data augmentation (jittering, scaling cho time series)
- Outlier removal (n_std=3)
- Label smoothing (optional)

**2. Model-Centric (Priority 2):**
- Early stopping (patience=15 cho LSTM, patience=10 cho TimesFM)
- L2 regularization (weight_decay=1e-5 cho LSTM, 1e-4 cho TimesFM)
- Dropout (0.2 cho LSTM layers, 0.3 cho FC layers)
- Layer normalization
- Learning rate scheduling (ReduceLROnPlateau)
- Gradient clipping (max_norm=1.0)

**3. Architecture-Specific:**
- **LSTM:** Recurrent dropout (built-in PyTorch), spatial dropout
- **GNN:** DropEdge (edge_drop=0.3), node dropout (0.2)
- **TimesFM:** LoRA dropout (0.1), gradient clipping

**Mandatory Checklist:**
```python
# Before Training
[ ] Data augmentation applied (if dataset < 5000)
[ ] Outliers removed (n_std=3)
[ ] Temporal split verified (NOT random)
[ ] Early stopping configured (patience=15)
[ ] Weight decay set (1e-5)
[ ] Dropout configured (0.2)
[ ] LR scheduler configured
[ ] Gradient clipping enabled

# During Training
[ ] Learning curves plotted every 10 epochs
[ ] Val loss monitored for overfitting signs
[ ] Checkpoints saved at best val loss

# After Training
[ ] Val-test metrics gap computed (< 0.05)
[ ] All 6 metrics evaluated
[ ] Results compared to baseline
```

**Documentation:** `docs/project/OVERFITTING_PREVENTION.md`

**Quick Implementation:**
```python
# Complete training loop with all anti-overfitting techniques
# See: docs/project/OVERFITTING_PREVENTION.md section 9
```

---

## 4. Model Architecture

### **Baseline Models**

#### **1. HAR-R Linear**
- **Features:** HAR (daily, weekly, monthly)
- **Method:** Linear regression
- **Purpose:** Baseline comparison
- **File:** `src/har_baseline/train.py`

#### **2. Simple LSTM**
- **Features:** Raw Parkinson volatility (1 input)
- **Architecture:** 1-layer LSTM, hidden_size=128
- **Purpose:** Deep learning baseline
- **File:** `src/lstm_baseline/train.py`

#### **3. LSTM-HAR**
- **Features:** HAR (daily, weekly, monthly) (3 inputs)
- **Architecture:** 2-layer LSTM, hidden_size=64
- **Purpose:** HAR + Deep learning
- **File:** `src/lstm_har_baseline/train.py`

#### **4. Enhanced LSTM-HAR**
- **Features:** Raw + HAR (weekly, monthly) (3 inputs)
- **Architecture:** 2-layer LSTM, hidden_size=64
- **Enhancement:** Raw volatility adds current-day info
- **Purpose:** Best performer (67.90% Dir Acc)
- **File:** `src/lstm_har_enhanced/train_enhanced.py`

### **Advanced Architecture: LSTM-GAT Hybrid** 🚀

#### **5. Temporal Graph Attention Network (TemporalGAT)** - NEXT GENERATION
- **Features:** 22 features (HAR + technical) for all 30 stocks simultaneously
- **Architecture:** LSTM (temporal) + Graph Attention Network (spatial)
- **Innovation:** Dynamic graph construction + multi-head attention
- **Target:** RMSE < 0.15, Dir Acc > 75% (vs current: 0.18, 67.90%)
- **Status:** Architecture design complete, ready for implementation
- **File:** `docs/project/LSTM_GAT_ARCHITECTURE.md`

**Key Components:**
1. **Per-Stock LSTM Encoder:** Temporal feature learning for each stock
2. **Dynamic Graph Builder:** Correlation + volatility spillover based edges
3. **Graph Attention Layers:** Multi-head attention for spatial relationships
4. **Temporal-Spatial Fusion:** Combines both branches for final prediction

**Performance Targets:**
- RMSE: 0.18 → **< 0.15** (17% improvement)
- Dir Acc: 67.90% → **> 75%** (7% improvement)
- QLIKE: ~0.12 → **< 0.10** (17% improvement)
- R²: ~0.65 → **> 0.75** (15% improvement)

**Advantages over LSTM-only:**
- ✅ Captures cross-stock correlations and spillover effects
- ✅ Dynamic graph adapts to changing market conditions
- ✅ Attention mechanism learns influential stocks
- ✅ Multi-scale: temporal (LSTM) + spatial (GAT)

**Implementation Roadmap:**
- Week 1: Data preparation (technical indicators, graph utilities)
- Week 2: Model development (LSTM encoder, GAT layers, fusion)
- Week 3: Training & evaluation (hyperparameter tuning, comparison)
- Week 4: Analysis & deployment (attention visualization, ablation)

#### **6. TimesFM 2.5 LoRA Fine-Tuning** - FOUNDATION MODEL APPROACH
- **Features:** Parkinson volatility (univariate time series)
- **Architecture:** TimesFM 2.5 (232M params) + LoRA adapters (~1.4M trainable params, 0.6%)
- **Method:** Decoder-only transformer with LoRA fine-tuning
- **Purpose:** State-of-the-art foundation model for time series
- **File:** `src/timesfm_baseline/timesfm_lora_finetuning.py`
- **Status:** Implementation complete, tested, reviewed
- **Documentation:** See `docs/timesfm/` for architecture and implementation details

**Key Innovations:**
- ✅ Foundation model approach (pre-trained on massive time series data)
- ✅ Parameter-efficient fine-tuning (LoRA adapters)
- ✅ Random window sampling (data-efficient training)
- ✅ No external normalization (TimesFM handles RevIN internally)
- ✅ Comprehensive testing (34 tests, 100% pass rate)

**Performance Targets:**
- RMSE: < 0.18 (baseline) → **< 0.15** (target)
- Dir Acc: > 55% (baseline) → **> 60%** (target)
- Training time: ~2 hours on GPU (vs ~30 min for LSTM)
- Trainable params: 1.4M (vs 65K for LSTM-HAR)

**Lessons Learned:**
- ⚠️ **3 adversarial reviews conducted** - Found 40 bugs total
- 📚 **Comprehensive lessons learned documented** - See `docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md`
- ✅ **Quick reference checklist created** - See `docs/QUICK_REFERENCE_CHECKLIST.md`
- 🔍 **All bugs fixed and tested** - 34/34 tests passing

---

## 5. Adversarial Review Process & Lessons Learned

This project uses adversarial code reviews to ensure production-ready code quality.

### **Adversarial Review Process**
1. **Cynical review** - Assume code has problems, look for hidden bugs
2. **Find 10+ issues** - Minimum threshold for review depth
3. **Fix all issues** - No exceptions, all HIGH/MEDIUM must be fixed
4. **Add unit tests** - Every fix must be tested
5. **Document lessons** - Add to knowledge base

### **TimesFM LoRA Review Results**
- **Review 1:** 15 bugs found (3 HIGH, 9 MEDIUM, 3 LOW)
- **Review 2:** 12 bugs found (3 HIGH, 6 MEDIUM, 3 LOW)
- **Review 3:** 13 bugs found (4 HIGH, 6 MEDIUM, 3 LOW)
- **Total:** 40 bugs fixed across 3 reviews

### **Key Lessons Learned Documents**
- 📘 **[Full Lessons Learned](docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md)** - Comprehensive guide with anti-patterns and mandatory practices
- 📋 **[Quick Reference Checklist](docs/QUICK_REFERENCE_CHECKLIST.md)** - Fast checklist for code reviews (87 items)
- 📊 **[Bug Statistics](docs/LESSONS_LEARNED_TIMESFM_ADVERSARIAL_REVIEWS.md#-summary-quick-reference)** - Breakdown by category and severity

### **Top 5 Bug Categories**
1. **Memory Management** (7 bugs) - Memory leaks, unbounded growth, improper cleanup
2. **Input Validation** (6 bugs) - Missing checks, late validation, poor error messages
3. **Data Pipeline** (8 bugs) - Silent data loss, inefficient tensor creation
4. **MLflow Integration** (4 bugs) - Metrics loss on crashes, poor error handling
5. **Resource Cleanup** (4 bugs) - File handles, model references, GPU memory

### **Mandatory Practices for Future Development**
- ✅ Validate all parameters at function entry (not deep in code)
- ✅ Create tensors on-the-fly (never pre-create in `__init__`)
- ✅ Use `pin_memory=True` + `non_blocking=True` for GPU training
- ✅ Save checkpoints BEFORE batch work (not after)
- ✅ Wrap MLflow calls in try/except per epoch
- ✅ Delete large objects ASAP (del + empty_cache)
- ✅ Test edge cases (empty, single, invalid inputs)
- ✅ Provide helpful error messages (what + why + how)

### **Quick Reference for Code Review**
Before approving any ML/DS code, verify:
- [ ] Data pipeline uses on-the-fly tensor creation
- [ ] GPU training uses `pin_memory=True` and `non_blocking=True`
- [ ] No silent data loss (warn if using `drop_last=True`)
- [ ] Checkpoints saved before work (not after)
- [ ] All parameters validated at entry point
- [ ] Error messages include what/why/how
- [ ] MLflow calls wrapped in try/except per epoch
- [ ] Large objects deleted explicitly (del + empty_cache)
- [ ] Tests for edge cases (empty, single, invalid)
- [ ] No memory leaks (profile long runs)

**See `docs/QUICK_REFERENCE_CHECKLIST.md` for full 87-item checklist.**

### **LSTM-GNN Normalization Failure (2026-06-21)**

**Issue:** Implemented dataset with `VolatilityNormalizer` but never used it in `__getitem__`, leading to:
- Trial-and-error with Softplus activation (collapsed predictions to 0)
- Wasted 3-4 hours debugging non-existent "negative prediction" problem
- Final solution: Just follow LSTM-HAR Enhanced's proven approach

**Root Cause:** 
- Didn't study LSTM-HAR Enhanced (67.90% Dir Acc) BEFORE implementing
- Assumed code worked without validation testing
- Trial-and-error instead of learning from proven solution

**LSTM-HAR Enhanced's Proven Approach:**
```python
# 1. StandardScaler normalization (mean=0, std=1)
target_scaler = StandardScaler()
target_scaler.fit(y_train)
y_train_norm = target_scaler.transform(y_train)

# 2. Linear output (NO activation like Softplus/ReLU)
output = self.fc(last_hidden_state)  # Can be negative on normalized scale

# 3. Inverse transform for evaluation
y_pred_original = target_scaler.inverse_transform(y_pred_norm)
# Now y_pred_original ≥ 0 (volatility is non-negative)
```

**Why This Works:**
- Normalized scale (mean=0, std=1) allows negative values during training
- Model learns patterns on normalized scale
- Inverse transform brings predictions back to physical scale (≥0)
- No need for activation functions to enforce non-negativity

**Mandatory Pre-Implementation Checklist (NEW):**
Before implementing ANY model architecture:
- [ ] Study existing successful implementation (e.g., LSTM-HAR Enhanced)
- [ ] Document their approach: normalization, activation, loss
- [ ] Identify patterns to follow vs patterns to improve
- [ ] Validate assumptions with simple tests (e.g., check if data is normalized)
- [ ] Compare architecture choices with proven solutions

**What We Should Have Done:**
```
Day 1 (BEFORE implementation):
  - Study: src/lstm_har_enhanced/model_enhanced.py
  - Document: "They use StandardScaler + linear output + inverse_transform"
  
Day 2:
  - Design LSTM-GNN following same pattern
  - Implement: StandardScaler in __getitem__, not just fit()
  - Implement: Inverse transform in validate()
  
Day 3:
  - Test and compare
  - Achieve: 64-65% Dir Acc immediately (no trial-and-error)
```

**What Actually Happened (Wrong Process):**
```
Day 1: Implement without studying reference
Day 2: Predictions negative → Add Softplus (wrong fix)
Day 3: Predictions collapse to 0 → Remove Softplus
Day 4: FINALLY check LSTM-HAR Enhanced → "They use StandardScaler!"
Day 5: Fix dataset to actually normalize → 64-65% Dir Acc
```

**Lesson:** Always study proven solutions BEFORE implementing, not after failures.

---

---

## 6. Evaluation Methodology

### **Current Status (CRITICAL BUG FOUND)**

**HAR-R Linear:** ✅ Uses temporal split (correct)  
**LSTM Models:** ❌ Use random split (DATA LEAKAGE!)

**Issue:** Random split allows future data in training → overestimated metrics

**Solution:** Implement 3-way temporal split (70/15/15)

**Files Created:**
- ✅ `src/common/temporal_split.py` - Temporal split utilities
- ✅ `src/lstm_har_enhanced/train_with_validation.py` - Example implementation
- ✅ `docs/project/TEMPORAL_SPLIT_EVALUATION.md` - Full documentation

---

## 6. Key Technical Decisions

### **Volatility Calculation**
- **Method:** Parkinson estimator
- **Formula:** σ² = (log(H/L)²) / (4*log(2))
- **Reason:** More efficient cho daily data than close-to-close

### **HAR Features**
- **Daily:** 1-day rolling mean
- **Weekly:** 5-day rolling mean
- **Monthly:** 22-day rolling mean (confirmed ~22 trading days/month)

### **Target Variable**
- **Horizon:** 5-day ahead (primary focus)
- **Method:** `volatility.shift(-5)`

### **Loss Functions**
- **Training:** MSE (convex, stable)
- **Evaluation:** QLIKE (academic standard) + RMSE, MAE, R², Dir Acc

### **Standard Hyperparameters (ALL Models)** ⭐

**CRITICAL:** All models MUST use these standardized hyperparameters for fair comparison.

**Training Configuration:**
```python
# ALL models (LSTM variants)
num_epochs = 70          # Maximum training epochs
patience = 15            # Early stopping patience
```

**Applied to All Files:**
- ✅ `src/lstm_har_enhanced/train_with_validation.py`
- ✅ `src/lstm_har_enhanced/train_enhanced.py`
- ✅ `src/lstm_har_baseline/train_with_validation.py`
- ✅ `src/lstm_har_baseline/train.py`
- ✅ `src/lstm_baseline/train_with_validation.py`
- ✅ `src/lstm_baseline/train.py`

**Why 70 epochs?**
- Sufficient for convergence without overfitting
- Allows early stopping to prevent overtraining
- Balances training time with model performance

**Why patience=15?**
- Gives model enough room to plateau before stopping
- Prevents premature stopping during temporary loss increases
- Standard practice for time series forecasting

---

## 7. Development Workflow

### **Code Review Process**

**Before Committing:**
- [ ] Variable names descriptive (not x, y, data)
- [ ] Functions small (< 30 lines)
- [ ] Docstrings cho public functions
- [ ] Tests added/updated
- [ ] Temporal split verified (not random)
- [ ] All 6 metrics calculated
- [ ] Learning curves plotted

**Before Merging:**
- [ ] All tests pass
- [ ] Coverage > 85%
- [ ] Documentation updated
- [ ] Results saved with timestamp

---

## 8. Documentation Structure

### **Key Documents**

**In Root (Only 3 .md files):**
- `README.md` - Project overview và quick start
- `CLAUDE.md` - This file (project rules)
- `project-context.md` - Project background

**In docs/:**
- `docs/project/` - Project-specific documentation
  - `TEMPORAL_SPLIT_EVALUATION.md` - Evaluation methodology
  - `OVERFITTING_PREVENTION.md` - Anti-overfitting techniques (MANDATORY)
  - `REFACTOR_SUMMARY.md` - Refactoring history
- `docs/lstm/` - LSTM model documentation
- `docs/common-rules/` - Reference to ml-ds-common-rules

**For detailed guides:** See respective directories in `docs/`

---

## 9. Quick Reference

### **Common Commands**

```bash
# Process data with outlier removal
python process_data.py --remove_outliers --n_std 3

# Train with all anti-overfitting techniques
python src/lstm_har_enhanced/train_with_validation.py \
    --dropout 0.2 \
    --weight_decay 1e-5 \
    --grad_clip 1.0 \
    --early_stopping_patience 15

# Train TimesFM with LoRA regularization
python src/timesfm_baseline/timesfm_lora_finetuning.py \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1

# Monitor overfitting during training
python check_overfitting.py --val_test_gap_threshold 0.05
```

### **File Locations**

```
Data:        data/processed/
Models:      models/*_2026-06-19_*/
Results:     results/*_2026-06-19_*/
Docs:        docs/
Source:      src/
```

---

## 10. Volatility Normalization Implementation (Project-Specific)

**For universal normalization best practices, see:**  
📘 **[Normalization Best Practices](D:\bmad-projects\ml-ds-common-rules\NORMALIZATION_BEST_PRACTICES.md)**

### **Current Implementation Status**

Following universal pattern from `ml-ds-common-rules`:

**File:** `src/lstm_gat_hybrid/dataset_with_graph_method.py`
- ✅ Uses `VolatilityNormalizer` (StandardScaler wrapper from `src/common/data_normalization.py`)
- ✅ Per-stock normalization (each stock has its own scaler)
- ✅ Proper transform in `__getitem__` (Line 280-316)
- ✅ Inverse transform in validation (Line 357-366 in `train_parallel_enhanced.py`)

**Training Results (Epoch 4/50):**
- Dir Acc: **64.87% - 65.05%** (Target: >67.90%)
- Predictions: NOT constant (variance = 0.042)
- Training: Stable, loss decreasing

### **Project-Specific Lessons Learned**

**LSTM-GNN Normalization Failure (2026-06-21):**

Timeline of what went wrong:
```
Day 1: Implemented dataset with VolatilityNormalizer
       - Initialized scalers (Line 182-186) ✅
       - But never used them in __getitem__ ❌

Day 2: Model predictions negative
       - Tried Softplus activation (wrong fix) ❌
       - Predictions collapsed to 0 ❌

Day 3: FINALLY checked LSTM-HAR Enhanced
       - Discovered: They use StandardScaler + linear output ✅
       - Fixed: Actually use scalers in __getitem__ ✅
```

**Root cause:** Didn't study proven implementation BEFORE coding.

**Resolution:** Fixed dataset to actually normalize (Line 280-316).

### **References**

- **Normalizer utility:** `src/common/data_normalization.py`
- **Universal patterns:** `ml-ds-common-rules/NORMALIZATION_BEST_PRACTICES.md`

---

## 11. Contact & Support

### **Getting Help**

**Questions:**
- Common rules: See `ml-ds-common-rules` package
- Project specifics: See `docs/project/`
- Integration: See `ml-ds-common-rules/INTEGRATION_GUIDE.md`

### **Project Status**

- **Phase:** LSTM-GNN Hybrid Development
- **Current:** Training Parallel LSTM-GNN (k-NN graph) - Epoch 4/50
- **Latest Results:** 64.87% - 65.05% Dir Acc (target: >67.90%)
- **Next:** Complete training, compare with baselines, analyze results

---

**Last Updated:** 2026-06-21
**Version:** 3.4 (Extracted Common Best Practices to ml-ds-common-rules)
**Status:** Active Development

---

**Changes in v3.4 (Current Version):**
- ✅ **Extracted normalization best practices** → `ml-ds-common-rules/NORMALIZATION_BEST_PRACTICES.md`
- ✅ **Reduced project-specific documentation** - Keep only volatility-specific lessons
- ✅ **Added universal patterns reference** - Link to common package for detailed guides
- ✅ **Cleaner separation of concerns** - Universal vs project-specific

**Changes in v3.3 (Previous Version):**
- ✅ **Added comprehensive normalization best practices** - Section 10
- ✅ **Documented LSTM-GNN normalization failure** - Lessons learned section
- ✅ **Added mandatory pre-implementation checklist** - Study proven solutions first
- ✅ **StandardScaler + linear output pattern** - Following LSTM-HAR Enhanced (67.90%)
- ✅ **Implementation template provided** - With validation tests
- ✅ **Common mistakes documented** - Anti-patterns to avoid

**Changes in v3.2 (Previous Version):**
- ✅ **Added mandatory overfitting prevention rules** cho ALL models
- ✅ **Created comprehensive anti-overfitting guide** - `docs/project/OVERFITTING_PREVENTION.md`
- ✅ **3 groups of techniques defined:** Data-centric, Model-centric, Architecture-specific
- ✅ **Mandatory checklist created** cho before/during/after training
- ✅ **Complete implementation examples** cho LSTM, GNN, TimesFM
- ✅ **Architecture-specific guidelines** added (Recurrent Dropout, DropEdge, LoRA)

**Changes in v3.1 (Previous Version):**
- ✅ **Standardized hyperparameters:** 70 epochs, 15 patience for ALL models
- ✅ **Added MSE to 6 mandatory metrics** (was 5, now 6)
- ✅ **Mandatory output format:** Console + JSON must include all 6 metrics
- ✅ **Updated all training files** to use standardized hyperparameters
- ✅ **Enhanced metrics reporting:** Added MSE to console, JSON, and comparison tables

**Changes in v3.0 (Previous Version):**
- ✅ Removed duplicated ML/DS common rules (now reference external package)
- ✅ Reduced from 1,798 lines to ~400 lines (58KB → ~12KB)
- ✅ Added 3-way temporal split evaluation section
- ✅ Added mandatory 6 metrics evaluation
- ✅ Streamlined to essential project-specific rules only
- ✅ Links to detailed docs in `docs/` instead of duplicating content
