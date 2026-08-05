# Gate ablation — per-ticker news gate vs. always-on news fusion (2026-08-05)

## Objective

Isolate the contribution of the per-ticker learned **gate** mechanism in the paper's news-fusion
headline model. The headline model `PerTickerGatedNewsBaseline`
(`baselines/2026-07-26_per_ticker_news_gate_baseline/`) gives each ticker its own independently
learned scalar sigmoid gate controlling how much that ticker relies on the news branch. The question:
does the learned per-ticker gate add value, or would simply concatenating news features WITHOUT any
gate (always fully using news for every ticker) perform just as well?

The sibling baseline `DualGroupNewsBaseline`
(`baselines/2026-07-25_dual_group_news_embedding_baseline/`) is architecturally identical EXCEPT it
has no gate — the exact no-gate counterpart. Per the gated baseline's own code comment, "the ONLY
architectural change is `PerTickerGatedNewsBaseline` (per-ticker gate) instead of
`DualGroupNewsBaseline` (no gate)." This is a clean single-variable ablation.

## Methodology

- **No-gate model rerun fresh** on the current P1.2-fixed pipeline (the old dual_group checkpoints
  are stale/invalid per `docs/reports/2026-08-03_canonical_results_table.md` — they predate the P1.2
  cross-stock date-alignment fix `6672ffa` and are not paper-comparable).
- **Same protocol as the gated reference numbers:** 3 seeds (42, 123, 2026), 20 epochs each, same
  inputs (`data/processed` 33 tickers + `data/features/dual_group_news_panel.parquet`), same
  dataloader (`create_dual_news_dataloaders`), same MSE loss on normalized scale, same 6-metric
  evaluation on denormalized scale, same corrected per-ticker DirAcc formula.
- **Gated reference numbers** taken from the already-confirmed post-P1.2 runs cited in
  `docs/reports/2026-08-03_final_paper_readiness_report.md` §1:
  `results/per_ticker_gate_2026-08-03_230821` (seed 42),
  `results/per_ticker_gate_2026-08-04_000448` (seed 123),
  `results/per_ticker_gate_2026-08-04_002252` (seed 2026).
- **No-gate result directories produced by this task:**
  `results/dual_group_news_2026-08-05_230040` (seed 42),
  `results/dual_group_news_2026-08-05_231746` (seed 123),
  `results/dual_group_news_2026-08-05_233438` (seed 2026). Each has `n_feat=146`, `d_news=64`,
  matching the dual_group architecture spec in the canonical table.
- Metrics read directly from each run's `results.json` → `test_metrics`. DirAcc column uses the
  corrected `directional_accuracy_per_stock` (equal to `directional_accuracy` in these runs, since
  the `n_stocks=` fix is applied — see code fix below).

## Code fix applied

`train_dual_news.py` already had the corrected per-ticker DirAcc fix
(`evaluate_predictions(..., n_stocks=n_stocks)` and a `directional_accuracy_per_stock` computation) —
no DirAcc change was needed. It had no epoch cap and no resume mechanism, so `--epochs 20` was passed
directly (this exact 20-epoch/3-seed protocol was authorized for this comparison).

The one change needed: the script hardcoded `torch.manual_seed(42)` / `np.random.seed(42)` with no
`--seed` CLI argument, so it could not run seeds 123/2026. A `--seed` argument was added (default 42,
preserving prior behavior), wired into both RNG seeds, and recorded in `results.json`, mirroring the
exact pattern already in `train_per_ticker_gate.py`. This is the only edit to the file.

Commit: `__COMMIT_SHA__` (filled in on commit).

## Results — test set, 3 seeds (42, 123, 2026), 20 epochs

### Mean ± std (n=3)

| Metric | No-gate (`DualGroupNewsBaseline`) | Gated (`PerTickerGatedNewsBaseline`) | HAR-only backbone (context)¹ |
|---|---|---|---|
| QLIKE ↓ | **0.4366 ± 0.0116** | 0.4430 ± 0.0185 | 0.4603 ± 0.0205 |
| RMSE ↓ | **0.002723 ± 0.000074** | 0.002734 ± 0.000096 | 0.002923 ± 0.000090 |
| MAE ↓ | **0.0007873 ± 0.0000100** | 0.0007930 ± 0.0000123 | 0.0008113 ± 0.0000137 |
| R² ↑ | **0.8047 ± 0.0106** | 0.8031 ± 0.0139 | 0.7749 ± 0.0140 |
| DirAcc (per-ticker) ↑ | **48.22% ± 0.27** | 47.77% ± 0.52 | 48.47% ± 0.35 |

¹ HAR-only backbone means from `docs/reports/2026-08-03_final_paper_readiness_report.md` §1, same
seeds/epochs/pipeline — shown for context only; it is the no-news reference, not part of the gate
ablation itself.

### Per-seed (matched seeds on both sides), diff = gated − no-gate

| Seed | Metric | No-gate | Gated | Diff (gate − no-gate) |
|---|---|---|---|---|
| 42 | QLIKE | 0.449687 | 0.464093 | +0.014406 (gate worse) |
| 42 | RMSE | 0.002803 | 0.002843 | +0.000040 (gate worse) |
| 42 | R² | 0.793221 | 0.787264 | −0.005957 (gate worse) |
| 42 | DirAcc | 47.9586 | 47.5594 | −0.399 (gate worse) |
| 123 | QLIKE | 0.432636 | 0.429440 | −0.003196 (gate better) |
| 123 | RMSE | 0.002709 | 0.002663 | −0.000046 (gate better) |
| 123 | R² | 0.806791 | 0.813272 | +0.006481 (gate better) |
| 123 | DirAcc | 48.2127 | 47.3780 | −0.835 (gate worse) |
| 2026 | QLIKE | 0.427607 | 0.435475 | +0.007869 (gate worse) |
| 2026 | RMSE | 0.002657 | 0.002695 | +0.000037 (gate worse) |
| 2026 | R² | 0.814081 | 0.808810 | −0.005270 (gate worse) |
| 2026 | DirAcc | 48.5030 | 48.3578 | −0.145 (gate worse) |

The sign of the gate's effect on QLIKE/RMSE/R² flips between seeds (gate helps on seed 123, hurts on
42 and 2026); on DirAcc the gate is lower on all three seeds but by a small margin.

### Paired t-test (gated − no-gate), n=3, df=2, two-tailed t_crit(0.05)=4.303

| Metric | Mean diff (gate − no-gate) | sd | t | Significant? |
|---|---|---|---|---|
| QLIKE | +0.006360 | 0.008897 | +1.238 | no |
| RMSE | +0.000011 | 0.000049 | +0.374 | no |
| MAE | +0.000006 | 0.000010 | +1.014 | no |
| R² | −0.001582 | 0.006991 | −0.392 | no |
| DirAcc | −0.459687 | 0.348722 | −2.283 | no |

No metric reaches significance at n=3 (all |t| < 4.303).

## Interpretation (for the Ablation Study subsection)

The per-ticker learned gate provides **no measurable benefit** over simple always-on news fusion. On
the mean across three seeds, the no-gate model (`DualGroupNewsBaseline`) is marginally better than the
gated model (`PerTickerGatedNewsBaseline`) on every one of the six metrics, but none of the
differences is statistically significant in a paired t-test (n=3, all |t| < 4.303), and the sign of
the gate's effect on the continuous-error metrics (QLIKE/RMSE/R²) is inconsistent across seeds. The
two architectures are, on this data and protocol, statistically indistinguishable, with at most a
slight non-significant edge to the simpler no-gate variant.

The practical implication for the paper: the improvement of news fusion over the HAR-only backbone
(significant on QLIKE/RMSE per the readiness report) comes from **the news features themselves, not
from the learned per-ticker gating**. Adding a per-ticker gate does not recover additional accuracy;
a simpler always-on-news fusion attains the same performance. By the simplicity principle, the
no-gate fusion is the preferable design unless the gate is retained for interpretability (the gate
values expose which tickers the model leans on news for, which the no-gate model cannot report).

Both news variants improve QLIKE/RMSE/R² over the HAR-only backbone and both have marginally lower
per-ticker DirAcc than HAR-only — consistent with the project-wide finding that news helps forecast
the magnitude of volatility, not the day-over-day direction.

**Sample-size caveat:** n=3 seeds is the minimum for a paired t-test; ≥5 seeds would strengthen the
"no difference" conclusion. The direction of the mean effect (no-gate ≥ gated on all six metrics) is
consistent enough that the qualitative conclusion — the gate does not add measurable value — is
robust to this caveat, but the exact magnitudes should be read as indicative, not definitive.

## Provenance

- No-gate runs: `results/dual_group_news_2026-08-05_230040` (seed 42),
  `.../231746` (seed 123), `.../233438` (seed 2026) — `results.json` records `seed`, `n_feat=146`,
  `d_news=64`, all 6 test metrics.
- Gated runs (reference): `results/per_ticker_gate_2026-08-03_230821` (seed 42),
  `.../per_ticker_gate_2026-08-04_000448` (seed 123), `.../per_ticker_gate_2026-08-04_002252`
  (seed 2026).
- Pipeline: post-P1.2 (`6672ffa`), corrected per-ticker DirAcc, split-first per-stock normalization.
