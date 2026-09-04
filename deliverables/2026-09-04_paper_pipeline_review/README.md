# Deliverables — paper pipeline for independent AI code review

Purpose: package the full volatility-forecasting pipeline that would back a conference/committee
paper submission, so an independent AI reviewer can audit it. **If the review passes, the paper is
drafted from these results.**

Scope (chosen 2026-09-04): the entire paper pipeline —
1. **Data**: ETL clean + enrich → `data/processed_enriched/{vn30,vn100}/` (causal, leakage-safe).
2. **VolGA walk-forward** (headline results): HAR / HAR-X / LSTM / VolGA on VN30 + VN100, multi-horizon
   {1,5,10,22}, 5 seeds, date-clustered Diebold–Mariano on 3 loss bases.
3. **Pooled/transfer VN30 ablation**: does widening the training universe (31→102) help VN30 (in
   progress — h1 done, h5–h22 running as of writing).

Review focus (all four): **leakage & correctness · reproducibility · paper-readiness · code + test
quality.** See `REVIEW_GUIDE.md`.

## How this package is organised

To avoid a second source of truth, this folder **points to the code in place** (the baselines are
already hard-isolated and self-contained) rather than copying it. Read the files at the paths listed
in `MANIFEST.md`. The evidence that is *derived* (results, claims, reproduce steps) is written out
here so the reviewer has the numbers without rerunning.

| File | What it gives the reviewer |
|---|---|
| `MANIFEST.md` | Exact paths of every code / config / test / data / result / report file, with purpose |
| `REVIEW_GUIDE.md` | The four review dimensions, each with concrete checks and where to look |
| `RESULTS_SUMMARY.md` | Headline numbers (QLIKE per model/market/horizon + DM p-values) + caveats |
| `CLAIMS.md` | The honest claims the paper would make, each with an evidence pointer |
| `REPRODUCE.md` | Exact commands to regenerate each result under the GPU venv |

## Status at packaging time (2026-09-04)

- VolGA walk-forward VN30 + VN100: **complete** (all 4 horizons, pushed).
- Pooled ablation: **h1 complete, h5–h22 running** (per-horizon JSONs are partial-safe).
- SP500 6-fold: queued/running (exploratory, not required for the core paper).
- All code pushed through commit `74a6441`+; results/reports for in-progress runs land under
  `results/…` and `docs/reports/…` as they finish.

## Non-negotiables the reviewer should assume

- Target = Parkinson **variance** (σ², not σ). Vietnamese raw prices are **not split-adjusted**
  (documented; see the overnight-tail appendix). QLIKE uses a positivity floor from the canonical
  config. These are disclosed limitations, not bugs — verify they are handled, not hidden.
- `archive/` is out of scope for review.
