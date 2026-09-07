# Code review — deep + HARQ stack baseline (2026-09-07)

Adversarial 3-lens review (subagent) of `code/deep_harq_stack.py` + `test/test_stack.py`, cross-checked
against the reused `harq_walkforward` (already reviewed), `run_masked_rich._pred_dict`, `wf_enriched_panel`,
and `stats.date_clustered_dm`.

## Verdict: NO CRITICAL, NO MAJOR. The "equal-weight stack beats HAR-X" claim is correctly wired.
Verified CLEAN (the 5 focus points):
1. **No test peeking:** `w_fit` is computed by `_fit_w` on val-only dicts (`recompute_harx` val split +
   `load_volga(...,'val')`); the primary weight is the a-priori 0.5. Test never influences either weight.
2. **Alignment consistent:** HAR dicts key `(panel.tickers[j], date)` (node-index→ticker via `_dict_by_ticker`);
   VolGA parquet keys `(ticker, date)`; both dates are the same `datetime_as_string(unit="D")` "YYYY-MM-DD",
   both panels built by the same `frozen_universe`+`build_enriched_panel(folds_target=7)`; `_align` intersects
   on (name,date); realized `y` taken uniformly from the HAR-X-Q dict → shared basis.
3. **Floor/orientation correct:** the applied floor is `training_config().qlike_floor` (identical to the
   reviewed HARQ baseline); `M.qlike` clamps y and p identically; `beats_HARX_sig = p<0.05 and mean_diff<0`
   is the correct DM orientation (stack = model A).
4. **No h/market mixup:** `load_volga` reads the matching `cells_{market}_..._h{horizon}` and filters
   `split ∈ {val,test}`, `model=='VolGA'`.
5. **Smoke non-trivial:** asserts stack < HAR-X AND < VolGA AND < HAR-X-Q AND `beats_HARX_sig` on real data.

## Findings and resolutions
| # | Sev | Finding | Resolution |
|---|-----|---------|------------|
| M-1 | MINOR (fail-loud) | `run()` test path had no non-empty/coverage guard on the 3-way merge — a stale VolGA dump could silently truncate the comparison set | **Fixed**: added `_require_overlap(len(y), len(vg_te))` (raises if the intersection is empty or <50% of VolGA test cells), unit-tested both branches. |
| M-2 | MINOR (dead const) | Module-level `FL = pc.QLIKE_FLOOR` was unused (the applied floor is `training_config`'s, correctly) | **Fixed**: deleted the dead `FL` line. |
| M-3 | MINOR (control only) | val-split `(ticker,date)` can repeat across folds; last-write-wins `.update()` gives an ambiguous val panel for `w_fit` | **Documented** in design.md; affects ONLY the val-fit overfitting control, not the a-priori 0.5 primary (test cells are disjoint across folds). |
| M-4 | MINOR (defensive) | `y` from HAR-X-Q dict not asserted equal to VolGA's own `y_true` | Not changed: `y_true` is deterministic per (ticker,date,horizon) so they agree by construction; noted as optional hardening. |

## Post-fix verification
- Tests 6/6 pass; **100% line + 100% branch** coverage on `code/`. ruff `--select F` clean.
- No CRITICAL/MAJOR; M-1/M-2 fixed, M-3 documented, M-4 accepted.
