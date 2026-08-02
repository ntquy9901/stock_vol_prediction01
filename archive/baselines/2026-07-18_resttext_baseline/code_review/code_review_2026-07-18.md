# Code Review — REST-TS Baseline (2026-07-18)

**Tool:** `/code-review`-style agent pass (1 agent, correctness focus across all 3 new
2026-07-18 baselines together) + self-fix, per CLAUDE.md DoD.

## Findings relevant to this baseline

- **REST-TS `.detach()` mechanism: CONFIRMED CORRECT.** Agent explicitly checked the residual
  detach is applied to the right tensor at the right point, with no gradient-leak path into
  `har_head`, and that `combined = har_pred + news_pred` is computed identically between
  `train_epoch` and `validate`. No fix needed.
- **Test coverage gap (CONFIRMED):** `test_smoke.py`'s residual-detach test re-implemented the
  loss logic inline instead of calling the real `train_resttext.py::train_epoch`. **Fixed:**
  added `test/test_train_loop.py`, which imports and runs the actual `train_epoch` on a tiny
  dummy `DataLoader`.

## Final state

4/4 pytest pass (3 smoke + 1 train-loop integration).
