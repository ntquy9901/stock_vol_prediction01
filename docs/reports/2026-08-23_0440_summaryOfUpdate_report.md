# Summary of update — overnight OOS run + two new paper versions (2026-08-23)

## Scope
Autonomous overnight mandate: (1) run 5 seeds × horizons {1,5,10,22} for HOSE+HNX, then a new paper
version; (2) run 5 seeds × {1,5,10,22} for S&P 500, then an even newer paper version. Both delivered.

## What changed
| Path | Purpose |
|---|---|
| `docs/paper/soict_harlstmgat_extended.tex` | Paper version 1 (goal 1): VN30/VN100 main + HOSE/HNX OOS section (committed 788a674, prior turn) |
| `docs/paper/soict_harlstmgat_crossmarket.tex` | Paper version 2 (goal 2): version 1 + "Cross-market OOS check: S&P 500" (metric table `tab:sp500`, DM table `tab:dmsp500`, interpretation); abstract + conclusion updated with the cross-market pattern. 12 pages. |
| `results/masked_rich_floor1e2/sp500_h{1,5,10,22}/result.json` | SP500 5-seed OOS results + validation-aligned GARCH row |

PDF artifacts (`*.pdf`) are gitignored per repo convention; only `.tex` is tracked. Commits: `788a674`
(extended), `5307ae6` (crossmarket + SP500 results).

## Results (5-seed means, main config: masked panel, z-score+linear, relative floor 1e-2·mean, --no-corr)

### SP500 (public OHLCV; 440 tickers after liquidity+history screen at h1; 714,520 obs / 1,624 test dates)
- HAR-X has the lowest QLIKE at every horizon; the gap is statistically significant (date-clustered DM):
  LSTM higher QLIKE than HAR-X at h1/h5/h10 (p<0.001, p<0.001, p=0.035); LSTM+GAT higher at h5/h10
  (p<0.001, p=0.002); neither significant at h22.
- Deep/graph advantage is on point error: LSTM+GAT lowest MSE/RMSE/MAE from h5 onward; MAE < HAR-X at
  h1/h5 (p=0.003); MAE < no-graph LSTM at every horizon (p<0.001). Graph lowers QLIKE vs LSTM only at h1
  (p<0.001).
- GARCH far worse (QLIKE ≈ 1.0, R² down to −38). R² negative at h22 for all models = long-horizon
  regression to the mean.

### Cross-market pattern (four VN panels + SP500)
Learned models cut squared and absolute error on the liquid large-cap universes (VN30, VN100, HOSE,
SP500) but lower QLIKE below HAR-X only on the less-liquid HNX exchange. Any deep/graph QLIKE advantage is
market- and liquidity-dependent, not universal. Honest, DM-verified wording throughout (no
"beats/wins/ties"; "no significant difference detected").

## Verification
- All 80 SP500 metric-table values checked programmatically against `result.json` → ALL MATCH.
- DM table: favours labels all present; 13 bold cells = 13 DM contrasts with p<0.05 (verified via awk dump).
- LaTeX: compiles clean (pdflatex ×2), 12 pages, no errors; no `\textbf`→TAB mangling (3 collapsed
  `\\` line-ends detected and fixed before final compile).

## Commands run
- `python scripts/garch_masked/run_oos_suite.py sp500` (GPU, background) → 4 cells DONE (~2500/2153/2239/1866 s)
- `pdflatex -interaction=nonstopmode soict_harlstmgat_crossmarket.tex` ×2 → 12-page PDF
- Numeric + DM verification scripts (inline) → all pass
- `git push origin master` → `5307ae6`, pre-push quality gate passed

## Code review
No source-code change this task (paper `.tex` + data `result.json` outputs from the already-reviewed
`run_oos_suite.py` / `run_masked_rich.py` harness). The runner and stats code were adversarially reviewed
earlier this session (Codex F1–F5 + rerun-02, all resolved). Verification here is data-integrity focused:
every reported number is traced to a stored `result.json`.

## Data-quality gate
N/A for this task (no data/feature/pipeline change — SP500 raw+processed already crawled, ETL'd and
Pandera-validated in the prior HOSE/HNX/SP500 data-quality pass, `2026-08-23_1030_hose_hnx_processed_quality.md`;
SP500 screen 498→440 tickers). Results are model outputs on already-validated processed data.

## Follow-ups
- Optional: `.md`/`.docx` companion export of the crossmarket paper (parity with the extended version) — not
  required by the mandate; deferred.
- Author decision: whether to fold the SP500 cross-market check into the submitted 8-page version or keep it
  as the discussion/extended track only.
