---
name: feedback-cross-project-vendoring
description: "When copying code from a sibling project that has its own train/test split assumptions baked into constants, recompute those constants against THIS project's actual split — don't copy the value as-is"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 7b3b1f97-cfdd-4b28-b9f4-b53d0110952d
  modified: 2026-07-24T18:32:51.683Z
---

When vendoring code from a sibling project (e.g. `C:\luanvan\data_eda`) that contains a
hardcoded split-boundary constant (a `TRAIN_CUTOFF`/`SPLIT_DATE`-style date used to fit something
like a PCA basis on "train-period-only" rows), do NOT copy that constant's value unchanged. The
source project's split boundary was chosen for ITS OWN split logic, which may differ completely
from the target project's split logic even when both look like an ordinary 70/15/15 chronological
split.

**Why:** In the `2026-07-25_dual_group_news_embedding_baseline` baseline, data_eda's
`TRAIN_CUTOFF="2020-01-01"` assumed one global calendar split date shared by every ticker.
`stock_vol_prediction01`'s actual split (`_split_raw_data_by_date`) instead cuts every ticker at
the **same row index** (not the same calendar date) — so each ticker's own val/test window lands
on a different date (earliest: 2010-06-30; latest: 2024-11-11 across VN30). Copying the
"2020-01-01" constant unchanged caused ~19/30 tickers' own val/test-period news rows to leak into
the "train" PCA fit — a real, confirmed data-leakage bug, only caught by manually walking each
ticker's actual split boundary and comparing against the copied constant.

**How to apply:** Before reusing ANY hardcoded split-boundary constant from a different project's
code, explicitly recompute the safe value against the CURRENT project's actual split function
(same ratios, same tickers) — take the minimum (earliest) per-entity split-boundary date across
all entities in scope, not just trust that "it's the same kind of split so the same date is
safe." This applies broadly, not just to this one baseline — any future cross-project code copy
that fits something on "pre-split-date" data needs the same check.
