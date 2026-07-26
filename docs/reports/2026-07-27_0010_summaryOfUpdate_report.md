# Summary — Fix: VPB/VRE Missing from News Ticker-Mention Regex

**What changed:** `baselines/2026-07-25_dual_group_news_embedding_baseline/code/vendor_config.py`
— added `"VPB", "VRE"` to `VN30_TICKERS` (the regex-source list used to tag which articles belong
to which ticker). Per user decision (2026-07-26 session): Issue A (32-vs-30 ticker universe stale
vs. current official VN30) deferred/documented only; Issue B (this fix) explicitly requested
("hãy fix bugs").

**Why:** `VN30_TICKERS` (30 entries) itself excluded VPB and VRE even though both are in this
project's own 32-ticker price universe (`data/processed/`). Verified before the fix:
`dual_group_news_panel.parquet` had zero rows for either ticker — their `x_news` input was an
all-zero vector in every news-fusion baseline built on this panel (dual_group, macro,
gated_crossattn, spillover_qlike, per_ticker_gate). Full root-cause trace: memory
`project_vn30_ticker_universe_mismatch.md`.

**Fix + rebuild:** added the 2 tickers, re-ran `build_dual_group_panel.py`. Result: **0 cache
misses** (the 2026-07-25 GPU `--include_all` expansion already covered these articles — no new
PhoBERT calls needed). Panel grew from 30×4890=146,700 rows to 32×4989=159,648 rows.

| Ticker | Before fix | After fix |
|---|---|---|
| VPB coverage | 0.00% (0 rows) | 61.54% |
| VRE coverage | 0.00% (0 rows) | 47.02% |

**Verification:** `pytest` across all 3 touched/dependent baselines (dual_group sibling,
spillover_qlike, per_ticker_gate) → 38/38 pass, no regressions.

**Not done (flagged for you):** no baseline has been re-trained on the fixed panel yet. Every
result reported earlier this session (dual-group retrain, spillover_qlike, per_ticker_gate
epochs 10/20/30/40) still reflects the PRE-fix panel — their VPB/VRE-specific numbers were noted
as artifacts to discard, but the OVERALL aggregate metrics for those runs are essentially
unaffected either way (VPB/VRE were just 2 of 30-32 tickers contributing all-zero news before).
A fresh re-run of `per_ticker_news_gate_baseline` (the current best/reference architecture) would
be needed to get a real, non-artifact VPB/VRE gate reading — not run yet, this session focused on
landing the fix + pushing to remote per your request.

## DoD checklist

- [x] Code satisfies the request (exact 2-ticker addition, no unrelated changes)
- [x] Tests run (38/38 pass across all dependent baselines)
- [ ] diff-cover — Not run (documented tooling gap)
- [x] Real-data verification (rebuilt panel, measured actual coverage before/after)
- [x] Impact analysis — traced all baselines that import `vendor_config.VN30_TICKERS`
      (read-only, single source of truth — no duplicated copies found)
- [x] Summary report — this file
