# Code Review — News Usefulness Ablation (2026-07-25)

**Method:** Self-directed adversarial review (consistent with the other baselines built today —
`/code-review` expects a GitHub PR, not applicable to uncommitted local work).

## Findings

### HIGH — Epoch-mismatch confound in the first ablation run (CONFIRMED, FIXED)

**File:** `code/compute_ablation_deltas.py` (input selection in `eval_checkpoint_per_ticker.py`)

The first delta computation compared the **40-epoch** (fully converged, early-stopped at epoch
36) all-ON dual-group checkpoint against a **10-epoch** HAR-only reference. Since the all-ON
model had ~4x more training, nearly every ticker (26/32) showed a QLIKE/MSE "improvement" —
plausibly reflecting "trained longer" rather than "news helps this specific ticker," which would
have invalidated the entire premise of using this ablation for a per-ticker decision.

**Fix:** re-ran `eval_checkpoint_per_ticker.py --checkpoint models/dual_group_news_2026-07-25_011719/best.pt`
(the 10-epoch all-ON checkpoint, matching HAR-only's own 10-epoch budget exactly) and
recomputed deltas. Result changed substantially: from 26 ON / 6 OFF to **11 ON / 21 OFF** — a
much more selective, plausible split. The requirements.md documents both numbers with the
before/after context for traceability.

### Notes, not findings (checked, no action needed)

- **`create_dual_news_dataloaders(news_panel_path=None)` gives identical train/val/test windows
  to a real-panel call** — verified: `x_har`/`adj`/`y` depend only on `_load_raw_stock_data`/
  `_split_raw_data_by_date`/`_generate_har_for_split`, none of which take the news panel as
  input; `common_stocks = sorted(set(...))` is deterministic. The HAR-only reference and the
  all-ON model are therefore evaluated on the exact same 32-ticker, same-window test set — a
  valid basis for per-ticker delta comparison.
- **QLIKE chosen as primary criterion, not DirAcc** — deliberate, given DirAcc's demonstrated
  instability at ~163 points/ticker in the two prior baselines today. The 11-ticker MSE agreement
  (11/11) vs. DirAcc agreement (4/11) in the final run itself reconfirms this choice was correct.
- **JSON float32 serialization bug** (both scripts, first run) — `evaluate_predictions()` returns
  numpy float32 for some keys; `json.dumps` can't serialize numpy scalars. Fixed by explicit
  `float()` casts in `per_stock_metrics`. Caught immediately (both scripts crashed at the final
  write step, output otherwise correct) — no results were lost since the fix just required a
  quick rerun.

## Summary

1 HIGH (epoch-mismatch confound) found and fixed before any downstream baseline consumed the
list — this is exactly the kind of dependency that could have quietly propagated a wrong
ticker classification into the next baseline if not caught here.
