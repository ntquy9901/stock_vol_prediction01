---
name: feedback-autonomous-execution
description: "User is comfortable with long autonomous multi-hour runs without per-step approval, once the plan/scope is agreed"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 7b3b1f97-cfdd-4b28-b9f4-b53d0110952d
  modified: 2026-07-24T18:32:40.297Z
---

When a task has an agreed plan (requirements.md/design.md written, or an explicit AskUserQuestion
round already resolved key decisions), and the user says something like "tôi đi ngủ, bạn cứ làm
hết, không cần chờ tôi approve bất cứ gì" — proceed through the ENTIRE remaining pipeline
autonomously: data copying, panel/feature rebuilding, training runs, test suite, code review, and
the summary report, without pausing for confirmation between steps.

**Why:** Said explicitly during the `2026-07-25_dual_group_news_embedding_baseline` session,
after the user had already resolved the key upfront decisions (data freshness, cross-project
copy strategy, model architecture pattern) via AskUserQuestion. They went to sleep and expected
to see finished results in the morning, not a half-done pipeline waiting on a checkpoint.

**How to apply:** Still pause and ask (even in this mode) when hitting a genuinely new decision
that wasn't covered by the earlier agreed plan and that the user would reasonably want to weigh
in on (e.g., whether to invoke PhoBERT at all — a previously-stated hard constraint — when 316
uncached articles turned up mid-run). Don't ask again about things already decided. If the user
sends a NEW instruction mid-run (e.g. "chạy thêm 10 epoch nữa xem có cải thiện không"), treat it
as pre-approval for that specific extension and fold it into the autonomous run — don't stop and
ask "should I do this now?".
