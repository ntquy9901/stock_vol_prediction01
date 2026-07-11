# Code Review — Decay baseline + recent un-reviewed changes (2026-07-11)

**Reviewer:** /code-review skill (high effort) — 3 finder agents (8 angles) + dedup/verify
**Scope (staged, 5 files):** `compute_decay.py`, `test_decay.py`, `extract_pilot_body.py`, `extract_embeddings.py` (mod), `aggregate_news_sources.py` (mod)

## Summary
- ~16 raw candidates → **10 survived**. Severity: **2 HIGH, 5 MEDIUM, 3 LOW**.

## Findings (most-severe first)

### [HIGH-1] compute_decay overwrites news_count_1d với 0/1 mask → confound so sánh
**File:** `compute_decay.py:48` · **Verdict:** CONFIRMED
Decay output ghi `news_count_1d = masks` (0/1). Original sentiment files có integer count (0,1,2...). Consumer `dataset_sentiment.py` scale count ×0.0005 → ngày 5 bài giờ thành `1×0.0005` thay `5×0.0005`. So sánh decay vs neutral-fill bị confound (2 biến đổi: decay + count).
**Fix:** Giữ nguyên news_count_1d gốc (integer) trong decay output, chỉ đổi sentiment_1d thành decayed state.

### [HIGH-2] Hardcoded `D:/bmad-projects/crawl_data/...` paths → vi phạm Code hygiene
**File:** `extract_embeddings.py:28`, `extract_pilot_body.py:14-16`, `aggregate_news_sources.py:24-25` · **Verdict:** CONFIRMED
CLAUDE.md Code hygiene: *"No hardcoded absolute local paths."* 3 files hardcode `D:/bmad-projects/crawl_data/...`.
**Fix:** Derive `CRAWL_DATA = _ROOT.parents[1] / "crawl_data"` (sibling convention, không hardcode绝对 path).

### [MED-3] compute_decay mask fallback `scores != 0` misclassifies neutral news
**File:** `compute_decay.py:71` · **Verdict:** PLAUSIBLE
Khi `news_count_1d` cột thiếu, fallback mask = `scores != 0`. Ngày có tin nhưng score=0.0 (neutral) → mask=0 → decay thay reset. (Thực tế sentiment files CÓ news_count_1d nên hiếm fire, nhưng fallback sai.)
**Fix:** Assert `news_count_1d` exists (input luôn có), bỏ fallback hoặc dùng.count thay !=0.

### [MED-4] extract_pilot_body silent "" khi fitz thiếu → 0 bodies, no error
**File:** `extract_pilot_body.py:30-32` · **Verdict:** PLAUSIBLE
Broad `except Exception: return ""` — nếu pymupdf chưa install, mọi PDF → "" silently, output 0 matches, exit success. User không biết thiếu dep.
**Fix:** Explicit `import fitz` ở top (fail loud nếu thiếu) thay try/except trong extract_body.

### [MED-5] aggregate on_bad_lines="skip" drops rows silently, no count
**File:** `aggregate_news_sources.py:42-44` · **Verdict:** PLAUSIBLE
Skip malformed rows (cafef line 3644...) không log số drop. Stats report undercount, không thấy data loss.
**Fix:** Đếm + log dropped rows trong stats (dùng callback hoặc engine=python counter).

### [MED-6] PCA fit calendar cutoff vs row-index split (leakage, inherited)
**File:** `extract_embeddings.py:130-137` · **Verdict:** PLAUSIBLE (inherited MED-4)
PCA fit `date<2020-01-01` nhưng split theo row-index 0.7 → xấp xỉ, weak leakage. Full fix = PCA-in-dataset post-split (defer, documented trước).
**Fix (this round):** document honest (đã note trong code).

### [MED-7] extract_embeddings content[:max_len*6] drops ticker deep in body
**File:** `extract_embeddings.py:99` · **Verdict:** PLAUSIBLE
`pat.findall(content)` chạy trên content ĐÃ truncate → ticker chỉ xuất hiện sâu trong body bị miss → article bị drop. --use_body mong đợi recall cao hơn nhưng ticker không ở lead thì bị loại.
**Fix:** Search ticker trên FULL content (trước truncate), encode trên truncated.

### [LOW-8] extract_embeddings vs extract_market_embeddings duplication ~80%
**Cleanup.** 2 script duplicate PhoBERT encode + PCA logic. Đã drift (market có PCA<2 guard, embedding không). Defer: isolation justifies; shared helper `baselines/_common/phobert_encode.py` là follow-up.

### [LOW-9] extract_pilot_body per-row df.at loop (vectorize được)
**Efficiency.** `for idx in df.index: df.at[idx,...]` chậm; vectorize via `.map`. Minor (21K rows ~2s).

### [LOW-10] extract_pilot_body join by basename fragile (collision)
**Altitude.** `body_map[p.name]` — nếu 2 source cùng pdf_filename → silent wrong join. Add collision warning (log distinct filename count).

## Remediation plan
- **Fix (HIGH+MED):** HIGH-1, HIGH-2, MED-3, MED-4, MED-5, MED-7 (6 fixes)
- **Document (defer):** MED-6 (PCA inherited), LOW-8/9/10 (cleanup)
- **Re-verify:** pytest + lint pass; diff-coverage đo

## Clean (verified)
- compute_decay formula đúng (test 5 pass), causal (carry-forward only).
- extract_pilot_body → extract_embeddings --use_body schema flow consistent (body column read đúng).
- aggregate dedup + 9-source unification đúng.
