# Summary of update — GARCH persistence cap (variance targeting) + re-run (2026-08-23)

## Trigger
Reviewer question: GARCH errors looked implausibly large (SP500 h1 MSE 57x HAR-X, R²=−37.7; HOSE h1
R²=−37.5), while VN30 was mild (R²=−0.11). An adversarial agent investigation
(`2026-08-23_0811_garch_bug_investigation.md`) confirmed **no code/units/scale/basis bug** (rebuilt panel
reproduced the stored SP500 MSE to 4 sig figs; pseudo-return scale exact). The magnitude was driven by ~5
near-unit-root (IGARCH, α+β=1.0 exactly) tickers whose frozen multi-step conditional-variance forecast
**diverges ~linearly** over the long (n_va+n_te, up to ~1800-step) forecast path, dominating the pooled MSE
(top-5 tickers = 69% of SSE). Decision (user): cap persistence + re-run for a more standard, defensible
benchmark.

## What changed
| Path | Change |
|---|---|
| `submission/soict_lstm_gat/baselines.py` | `garch_forecast` now builds the analytic multi-step variance path with persistence capped at 0.999 via **variance targeting** (`_cap_params`, `_capped_forecast_path`). For a stationary fit this reproduces arch's analytic forecast (VN30 unchanged); a near-integrated fit converges to a finite capped unconditional variance instead of diverging. |
| `submission/soict_lstm_gat/tests/test_baselines.py` | +4 TDD tests: cap reduces persistence via variance targeting, no-op when stationary, analytic path matches direct recursion when stationary, IGARCH path stays bounded/converging (not diverging). |
| `results/masked_rich_floor1e2/{vn30,vn100,hose,hnx,sp500}_h{1,5,10,22}/result.json` | 20 cells: `metrics['GARCH']` + `dm_date_clustered['GARCH_vs_HARX']` recomputed with the capped forecast. |
| `docs/paper/soict_harlstmgat{,_extended,_crossmarket,_with_sp500}.tex` | GARCH table rows updated (VN100/HOSE/HNX/SP500 changed; VN30 unchanged); GARCH method sentence discloses the persistence cap; mechanism prose corrected (removed the inaccurate "converges … level shifts" / "illiquid series" wording — the near-integrated tickers diverged, they are not illiquid). |

## Effect (capped vs old GARCH; HAR-X context unchanged)
| Panel h1 | old MSE(×1e7) | old R² | new MSE | new R² |
|---|---|---|---|---|
| VN30  | 2.788  | −0.113 | 2.788 (unchanged) | −0.113 |
| VN100 | 15.571 | −4.108 | 3.816  | −0.252 |
| HOSE  | 156.06 | −37.53 | 18.495 | −3.566 |
| HNX   | 14.0*  | −7.0*  | 21.336 | −0.197 |
| SP500 | 351.91 | −37.68 | 13.619 | −0.497 |

GARCH still has a significantly higher QLIKE than HAR-X in every cell (date-clustered DM p<0.001, except
VN30 h22 p=0.001) and a negative R² everywhere — it remains a clearly-dominated classical benchmark, now
without the near-integrated divergence artifact. VN30 is bit-for-bit unchanged (its fits are stationary, the
cap never binds).

## Verification
- Unit tests: `test_baselines.py` 8/8 pass; `scripts/garch_masked/test_garch_masked.py` 4/4 pass.
- Empirical direction check before the full re-run: SP500 h1 MSE 351.9→13.6, R² −37.7→−0.50; VN30 h1 identical.
- Re-run all 20 cells (`garch_recompute_capped.log`); each cell passes the HAR-X basis guard
  (recomputed HAR-X QLIKE matches the stored value before GARCH is overwritten).
- Paper consistency: 260/260 GARCH table cells across the four `.tex` files match the recomputed
  `result.json`. All four papers compile clean (pdflatex ×2): 8 / 10 / 12 / 10 pages, 0 errors.

## Code review
Change is localized to `garch_forecast` + two pure helpers, covered by TDD tests (property + recursion-match +
IGARCH-bound). The analytic path equals arch's forecast on stationary fits, so non-near-integrated results are
unchanged; the only behavioral change is bounding near-integrated divergence, which is the intended fix. No
leakage introduced (fit remains train-only; cap uses the train-series sample variance as the targeting level).

## Data-quality gate
N/A (no data/feature/pipeline change — only the GARCH benchmark computation and its stored metrics). Processed
+ raw data for all five panels are already Pandera-validated (`2026-08-23_1030_hose_hnx_processed_quality.md`).

## Follow-ups
- Deliverables package `deliverables_20260823` to be refreshed with the corrected papers + recomputed
  result.json + capped `baselines.py` + this report before zipping for review.
