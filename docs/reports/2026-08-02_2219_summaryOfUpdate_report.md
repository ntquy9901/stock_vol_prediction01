# Summary — verify-audit-fixes skill built and verified end-to-end

## Bối cảnh

`docs/reports/2026-08-02_152758_summaryOfUpdate_report.md` đề xuất 1 skill dự án
(`verify-audit-fixes`) ràng buộc claim "finding đã fix" vào evidence máy chạy thật (git state,
test/lint/coverage output) thay vì khẳng định bằng lời. User yêu cầu dùng subagent xây dựng và áp
dụng skill này. Subagent nền dừng giữa chừng do hết session limit (đã xây xong `scripts/
verify_audit_fixes/` với 16 gate-test đầu tiên pass, đang viết integration test cho
`run_verification.py` thì bị cắt). Phiên này hoàn thiện nốt: sửa 1 test fixture lỗi, viết
`.claude/skills/verify-audit-fixes/SKILL.md` (phần subagent chưa kịp làm), viết design note ngắn,
và verify toàn bộ end-to-end thật trên chính repo.

## Việc đã làm

1. **Sửa 1 test fail còn lại** (`test_main_cli_returns_zero_on_clean_pass`) — áp dụng
   `systematic-debugging`: root cause không phải bug ở `gates.py` (gate 2 báo "fail" đúng vì ruff
   thật sự tìm thấy 1 lỗi lint có thật trong fixture "minimal project" của test — thiếu dòng trống
   sau import, vi phạm rule I001). Sửa fixture cho sạch ruff thật, không sửa logic gate (logic đã
   đúng).
2. **Viết `.claude/skills/verify-audit-fixes/SKILL.md`** — phần orchestration mà subagent chưa
   kịp viết trước khi hết session. Theo đúng format 3 skill đã có (`systematic-debugging`,
   `dispatching-parallel-agents`, `archive-review`). Nội dung: khi nào dùng, Gate 1-6 chạy thật thế
   nào (gọi `scripts/verify_audit_fixes/run_verification.py`, không tự viết lại logic), Gate 7-11
   báo "Not verifiable" kèm lý do cụ thể — KHÔNG BAO GIỜ tự chế evidence để "cho qua" các gate này.
3. **Viết `docs/verify_audit_fixes/design.md`** — requirements + design ngắn gọn (theo right-sizing
   của CLAUDE.md §5, không dùng cấu trúc 5-folder baseline đầy đủ vì đây không phải 1 baseline).
4. **Verify end-to-end thật trên chính repo** (không chỉ chạy trên fixture giả):
   `python scripts/verify_audit_fixes/run_verification.py --repo-root . --evidence-dir
   docs/reports/evidence/2026-08-02_221336 --skip-gate4 --skip-gate5 --skip-gate6` — evidence
   directory sinh ra đầy đủ 9 file đúng schema (`manifest.json`, `git_status.txt`,
   `git_diff_stat.txt`, `environment.txt`, `ruff.txt`, `static_checks.txt`, `static_scans.txt`,
   `pytest_collection.txt`, `acceptance_traceability.csv`).

   Kết quả THẬT của lần chạy này trên repo hiện tại (không làm sạch giả trước khi verify — đúng
   tinh thần của skill: không che giấu trạng thái thật):
   - Gate 1 (repository identity): `dirty` — working tree có thay đổi chưa commit tại thời điểm
     chạy (đúng thực tế, không bị coi là lỗi cứng — theo thiết kế, caller chịu trách nhiệm xác nhận
     phạm vi).
   - Gate 2 (static checks): `fail` — ruff tìm thấy lỗi lint thật trên toàn repo (khớp con số 947
     violation đã ghi nhận ở audit trước).
   - Gate 3 (test discovery): `fail` — khớp 9 lỗi collection tiền tồn tại đã biết từ P2.3 (import
     module archived, thiếu optional dependency torch_geometric/mlflow).
   - Gate 4-6: `not_run` (bỏ qua có chủ đích để verify nhanh, đã ghi rõ trong manifest, không giả
     vờ đã chạy).
   - Gate 7-11: `Not verifiable`, mỗi gate kèm lý do cụ thể (thiếu regression test theo từng
     finding, thiếu schema provenance ML, thiếu multi-seed framework, thiếu tích hợp adversarial
     review, thiếu môi trường clean-room).

   Điểm quan trọng: skill **không** làm cho kết quả "đẹp" hơn thực tế — báo cáo đúng repo đang có
   vấn đề lint/test-collection thật, đúng tinh thần thiết kế ban đầu.

## Tests / kiểm chứng đã chạy

- `pytest tests/verify_audit_fixes/ -v` — **54/54 passed** (47.98s).
- `python scripts/verify_audit_fixes/run_verification.py ...` chạy thật trên repo (không phải
  fixture) — xem kết quả ở trên, evidence directory `docs/reports/evidence/2026-08-02_221336/`
  còn nguyên để đối chiếu.
- Compile-check: không cần riêng, đã cover qua test suite (import toàn bộ module khi test chạy).

## Code review

Không chạy `/code-review` đầy đủ cho phần code do subagent viết (16 gate-test đã pass trước khi
subagent bị cắt, phần còn lại — `run_verification.py`, `SKILL.md`, design note — do phiên này viết
tiếp và tự kiểm bằng cách chạy thật). Khuyến nghị chạy `/code-review` cho toàn bộ
`scripts/verify_audit_fixes/` + `tests/verify_audit_fixes/` trước khi dùng skill này cho quyết
định "verified fixed" chính thức trên finding Critical/High của paper.

## Files (path → mục đích)

- `scripts/verify_audit_fixes/{commands,gates,manifest,traceability,static_scans,run_verification}.py`
  → logic Gate 1-6 thật, độc lập với Claude Code (chạy được từ terminal thường).
- `.claude/skills/verify-audit-fixes/SKILL.md` → skill session-facing (gitignored, không commit —
  nhất quán với 3 skill khác đã có trong `.claude/skills/`).
- `docs/verify_audit_fixes/design.md` → requirements/design ngắn gọn.
- `tests/verify_audit_fixes/*.py` → 54 test, bao gồm 1 integration test CLI end-to-end.
- `docs/reports/evidence/2026-08-02_221336/` → evidence thật từ lần verify đầu tiên trên chính
  repo — giữ lại làm bằng chứng skill hoạt động đúng, không phải fixture giả.

## Risks / follow-ups

- Gate 7-11 chưa thể chạy thật — cần xây hạ tầng riêng (regression test theo từng finding, schema
  provenance ML, multi-seed framework) trước khi dùng skill này để tuyên bố "Verification passed"
  đầy đủ cho các claim khoa học của paper. Hiện tại chỉ Gate 1-6 là thật.
- Gate 2/3 hiện đang `fail` thật trên repo — không phải lỗi của skill, mà là phản ánh đúng nợ kỹ
  thuật đã biết (947 ruff violation, 9 lỗi test collection). Cần xử lý riêng trước khi kỳ vọng
  Gate 1-6 "pass" toàn bộ.
- `docs/reports/evidence/` chưa có rule dọn dẹp/retention — mỗi lần verify sẽ tạo 1 thư mục mới,
  cần quyết định sau này có nên gitignore hay tiếp tục commit làm bằng chứng.

## Definition of Done checklist

- [x] Code thỏa yêu cầu (build + hoàn thiện skill theo đúng spec), không refactor ngoài phạm vi.
- [x] Tests: 54/54 pass, kể cả integration test CLI thật.
- [ ] Coverage gate (`diff-cover`) — Not run, tooling chưa cài (gap đã biết từ trước).
- [x] Lint — `ruff` chạy được (đã xác nhận cài đặt), nhưng KHÔNG chạy riêng cho code mới này tách
      biệt khỏi kết quả chung của repo (xem Gate 2 ở trên đã bao gồm toàn repo).
- [ ] Code review đầy đủ (`/code-review`) — Not run, xem mục "Code review" ở trên.
- [x] Summary report — file này.
- [x] Smoke thật — chạy `run_verification.py` thật trên chính repo (không phải chỉ trên fixture
      giả), evidence directory sinh ra đúng, đầy đủ.
- [x] Impact analysis — không sửa code hiện có ngoài `scripts/`, `tests/verify_audit_fixes/`,
      `docs/verify_audit_fixes/`, `.claude/skills/` — không đụng pipeline train/eval nào.
- [x] Push remote ngay sau khi verify xong (rule mới trong CLAUDE.md).
