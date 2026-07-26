# Code Review — Selective News Gate Baseline (2026-07-25)

**Method:** Self-directed adversarial review (same reasoning as the sibling
`2026-07-25_dual_group_news_embedding_baseline`'s review — `/code-review`'s automated tooling
expects a GitHub PR, not applicable to this uncommitted local work).

## Findings

### MEDIUM — Ticker universe mismatch (CONFIRMED, FIXED)

**File:** `code/model_selective_gate.py`

The EDA report (`docs/suggestion/2026-07-25_professor_report.md`) classified 30 VN30 tickers.
The actual training pipeline's stock universe (`_load_raw_stock_data`/`_split_raw_data_by_date`)
has **32 common stocks** — VPB and VRE are not in the EDA's 30-ticker list. `build_stock_mask`'s
fail-loud check (`ValueError` on unclassified tickers) caught this immediately on the first real
(non-smoke) run, rather than silently defaulting them to some guessed value.

**Fix:** VPB/VRE added to `NEWS_OFF_TICKERS` with an inline comment — no ΔR² evidence exists
either way for them, so the conservative default (OFF) is used, consistent with this baseline's
premise (only turn news ON where there's positive evidence).

### Notes, not findings (checked, no action needed)

- **Mask correctness verified two ways:** (1) unit test (`test_mask_correctness.py`) proves exact
  numerical equality of a NEWS_OFF stock's prediction regardless of its own news input; (2) the
  real training run's per-ticker breakdown shows NEWS_OFF tickers' predictions are driven purely
  by the (shared) HAR branch, consistent with the mask working as designed.
- **`stock_mask` is a buffer, not `nn.Parameter`** — confirmed via `test_forward_shape_and_backward`
  asserting `model.stock_mask.grad is None` after backward.
- **No changes to the sibling baseline** — `dataset_dual_news.py`/`model_dual_news.py` imported
  read-only via `sys.path` injection, matching the project's established isolation convention.

## Result finding (not a code bug — an empirical result)

The 10-epoch training run's own result is itself the main "finding" of this baseline: the
EDA-based ticker selection (from an unrelated HGB/XGBoost per-ticker model family) does **not**
transfer to this shared LSTM-GNN architecture — NEWS_OFF tickers scored a HIGHER average
per-ticker test DirAcc (51.60%) than NEWS_ON tickers (46.29%), the opposite of the hypothesis.
See the summary report for full discussion; this is a valid experimental outcome, not a defect to
fix.

## Summary

1 MEDIUM (ticker-universe mismatch) confirmed and fixed. All 6 tests pass. No correctness issues
found in the masking mechanism itself — the negative result is a genuine finding about whether
tree-model-derived per-ticker news usefulness transfers to this architecture, not a bug.
