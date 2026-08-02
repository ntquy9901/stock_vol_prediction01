# Code Review — Top-3 News Gate Baseline (2026-07-25)

**Method:** Self-directed adversarial review (same rationale as the two sibling baselines today —
`/code-review` expects a GitHub PR, not applicable here).

## Findings

No new correctness issues. This baseline is a narrow, mechanical variant of
`2026-07-25_selective_news_gate_baseline` (same mask-after-LSTM mechanism, already reviewed and
unit-tested there) — only the ticker allowlist changed (22-ticker classification -> strict
3-ticker allowlist {VIB, ACB, MWG}).

- **Buffer overwrite pattern verified safe:** `Top3NewsGateBaseline.__init__` calls
  `super().__init__()` (which registers `stock_mask` via `register_buffer`, using the PARENT's
  22/10 classification) then immediately overwrites `self.stock_mask` with the narrower 3-ticker
  mask. Confirmed via `nn.Module.__setattr__` semantics (re-assigning an already-registered
  buffer name updates `self._buffers`, not a shadow plain attribute) — verified empirically by
  `test_build_stock_mask_is_strict_allowlist` + the real run's own log line
  (`NEWS_ON stocks=3/32`).
- **4/4 tests pass**, including exact-zero-contribution checks for non-top-3 stocks (same
  numerical-equality style as the sibling baseline's test).

## Result finding (not a code bug — an empirical result)

Narrowing to the 3 tickers with the strongest, most consistent EDA signal did **not** produce a
clear positive result either. Val-set ON/OFF gap (52.35% vs 46.75%, +5.6pp) looked promising but
did **not replicate on the test set** (48.67% vs 48.89%, essentially a tie, -0.22pp in the
"wrong" direction). With only 3 tickers in the ON group, the average is highly sensitive to
per-ticker noise (test DirAcc for VIB/MWG/ACB individually: 53.37% / 47.24% / 45.40% — an 8pp
spread among just these 3). See summary report for full discussion.

## Summary

0 code findings. The experiment itself is inconclusive (neutral, not confirmatory) — reported
honestly rather than overstated as either a win or a repeat of the prior contradiction.
