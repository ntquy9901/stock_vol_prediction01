---
name: project-dual-group-news-baseline
description: "2026-07-25 baseline bringing data_eda's dual-group PhoBERT+EWMA news embeddings into stock_vol_prediction01, results and known limitations"
metadata:
  node_type: memory
  type: project
  originSessionId: 7b3b1f97-cfdd-4b28-b9f4-b53d0110952d
  modified: 2026-07-24T18:33:04.210Z
---

`baselines/2026-07-25_dual_group_news_embedding_baseline/` ports the dual-group
(khach_quan/tong_hop source split) PhoBERT→PCA(32)→EWMA(30d) news-embedding pipeline from the
sibling project `C:\luanvan\data_eda` into this project, as a new baseline compared against
`2026-07-07_embedding_baseline` (single-group PCA-64, 68.76% DirAcc @40ep).

**Result (post-leakage-fix, the correct numbers):** 10 epochs → Val DirAcc 69.68% / Test 68.50%.
20 epochs → Val 70.00% / Test 68.25% — essentially no improvement from training longer; results
plateau around epoch 9-10. Comparable to or slightly better than the original PCA-64 baseline, in
4x fewer epochs.

**Why:** User wanted to know whether data_eda's richer dual-group+EWMA feature set (not yet used
anywhere in stock_vol_prediction01) beats the simpler PCA-64 embedding already used by 6+ other
news baselines in this project.

**How to apply:** If asked to extend or compare against this baseline, use the POST-FIX numbers
above (results/dual_group_news_2026-07-25_0117* and *_0122*), not the earlier same-day runs
(results/dual_group_news_2026-07-25_0036*/*_0046* are pre-leakage-fix and documented as
superseded in the summary report). See [[feedback-cross-project-vendoring]] for the leakage bug
that was found and fixed mid-session. Full writeup:
`docs/reports/2026-07-25_0131_summaryOfUpdate_report.md`.

Known limitation: PCA basis fit only on pre-2010-06-30 news (~4 years) due to the leakage fix,
much less than data_eda's original ~14-year window — a genuinely ticker-aware (per-ticker cutoff)
PCA fit would recover more training data but wasn't implemented (shared-PCA design assumes one
global cutoff).
