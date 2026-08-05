# Archive batch log — 2026-08-02

Restoration reference for the archival batch executed 2026-08-02, based on cross-referencing
`docs/report_2026-06-27/`, `docs/report_2026-07-11/`, `docs/report_2026-07-18/`,
`docs/report_2026-07-25/`, `docs/report_2026-08-01/` against current codebase state (5 parallel
investigation passes, one per report — see `docs/reports/` for the individual findings this
batch executed on). Pre-batch `HEAD`: `9332cd01a9bf9ce976c760115d489bbc309f3da9`.

Every move used `git mv` (history preserved) except the 3 already-gitignored `data/` folders,
which used a plain `mv` — restoring those does not need git, just moving the directory back.

## To restore anything below

- **Git-tracked moves:** `git mv archive/<new_path> <old_path>` (or `git revert` the commit(s)
  this log's own commit created, listed at the bottom once committed).
- **Gitignored `data/` moves:** `mv archive/data/<name> data/<name>`.

## Baselines → `archive/baselines/`

| Original path | Archived path |
|---|---|
| `baselines/2026-07-07_embedding_baseline` | `archive/baselines/2026-07-07_embedding_baseline` |
| `baselines/2026-07-08_market_fallback` | `archive/baselines/2026-07-08_market_fallback` |
| `baselines/2026-07-11_sentiment_decay` | `archive/baselines/2026-07-11_sentiment_decay` |
| `baselines/2026-07-11_latent_noise` | `archive/baselines/2026-07-11_latent_noise` |
| `baselines/2026-07-11_sentiment_price_eda` | `archive/baselines/2026-07-11_sentiment_price_eda` |
| `baselines/2026-07-15_pure_market_baseline` | `archive/baselines/2026-07-15_pure_market_baseline` |
| `baselines/2026-07-18_alignment_loss_baseline` | `archive/baselines/2026-07-18_alignment_loss_baseline` |
| `baselines/2026-07-18_gated_crossattn_baseline` | `archive/baselines/2026-07-18_gated_crossattn_baseline` |
| `baselines/2026-07-25_selective_news_gate_baseline` | `archive/baselines/2026-07-25_selective_news_gate_baseline` |
| `baselines/2026-07-25_top3_news_gate_baseline` | `archive/baselines/2026-07-25_top3_news_gate_baseline` |
| `baselines/2026-07-25_news_usefulness_ablation` | `archive/baselines/2026-07-25_news_usefulness_ablation` |
| `baselines/2026-07-25_ablation_derived_gate_baseline` | `archive/baselines/2026-07-25_ablation_derived_gate_baseline` |

Reason (each): confirmed null/rejected/inconclusive result per the baseline's own
`requirements.md` go/no-go AND per the corresponding weekly report's own stated conclusion, AND
zero remaining live importers (verified via repo-wide grep, excluding comment/docstring-only
mentions). `2026-07-07_embedding_baseline` and `2026-07-15_pure_market_baseline` were archived as
part of this same batch specifically because their only real code consumers
(`alignment_loss_baseline`, `gated_crossattn_baseline`, `latent_noise`, `pure_market_baseline`
importing from `embedding_baseline`) are all in this same batch — archiving the dependency without
its dependents (or vice versa) would have stranded one side.

**Not archived, checked and confirmed still load-bearing:** `2026-07-25_dual_group_news_embedding_baseline`
(its own result was also null, but it's a live read-only dependency of
`2026-07-26_per_ticker_news_gate_baseline`, `2026-07-26_spillover_qlike_baseline`, and
`2026-08-01_calendar_news_gate_baseline` — confirmed via `_SIBLING_CODE` sys.path injection in
those baselines' code, not just comment mentions). `2026-07-25_expand_news_cache_baseline`,
`2026-07-25_macro_news_baseline` — not covered by any of the 5 reports' own conclusions, left as-is
pending a future decision. `2026-07-15_objective_news_baseline` — explicitly flagged as needing a
human decision (finish vs. formally close) in an earlier pass, deliberately not resolved here.

## `src/` module → `archive/src_legacy/`

| Original path | Archived path |
|---|---|
| `src/sentiment_baseline` | `archive/src_legacy/sentiment_baseline` |

Reason: only real importers were `2026-07-11_sentiment_decay` and `2026-07-11_sentiment_price_eda`
(both archived in this same batch, above) — confirmed via grep; the one other repo-wide
`sentiment_baseline` string hit (`src/data_aggregation/aggregate_news_sources.py`) is a comment
stating explicit non-interaction, not an import.

## Documentation snapshots → `archive/docs_reports_legacy/`

| Original path | Archived path |
|---|---|
| `docs/report_2026-06-27/` (entire folder, incl. `04_code/` stale copies of `dataset_with_graph_method.py`/`model_parallel.py`/`train_parallel_enhanced.py` and `.pth` checkpoints) | `archive/docs_reports_legacy/report_2026-06-27/` |
| `docs/report_2026-07-11/` | `archive/docs_reports_legacy/report_2026-07-11/` |
| `docs/report_2026-07-18/` | `archive/docs_reports_legacy/report_2026-07-18/` |

Reason: zero references from any current `.py` file or from the current main report
(`docs/report_2026-08-01/BAO_CAO_TONG_HOP.md`), confirmed via repo-wide grep for each folder's own
path string. `docs/report_2026-06-27/02_technical_docs/{PARALLEL_LSTM_GNN_ARCHITECTURE.md,PAPER_ANALYSIS_SONANI_2025.md}`
were confirmed byte-identical duplicates of the still-live `docs/project/` copies before archiving
(`diff` returned no output for both pairs).

**Not archived:** `docs/report_2026-07-25/` — still directly cited by
`docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` for specific numbers; `docs/report_2026-08-01/` — the
current main report.

Two dangling citations to the now-archived `docs/report_2026-06-27/` were patched (not left as dead
links) in the same batch:
- `docs/report_2026-07-25/BAO_CAO_CHO_THAY.md` §5 (line 235) — added a note pointing to the new
  archive location, table data below the citation left unchanged.
- `docs/project/SENTIMENT_ANALYSIS_DESIGN.md` (line 549) — repointed to
  `docs/report_2026-08-01/BAO_CAO_TONG_HOP.md` as the current model report, with a note on the old
  path's new location.

## Data → `archive/data/` (cascading orphans, discovered after the code moves above)

| Original path | Archived path | Why orphaned |
|---|---|---|
| `data/sentiment_embedding/` | `archive/data/sentiment_embedding/` | Its only live consumers (`market_fallback`, `latent_noise`, `pure_market_baseline`, `alignment_loss_baseline`, `gated_crossattn_baseline` `--emb_dir` defaults) are all archived above. |
| `data/sentiment_decay/` | `archive/data/sentiment_decay/` | Its only consumer (`baselines/2026-07-11_sentiment_decay/code/compute_decay.py`) is archived above. |
| `data/sentiment_baseline/` | `archive/data/sentiment_baseline/` | Its only consumers (`src/sentiment_baseline/process_news_to_sentiment.py`, archived above, and the also-archived `sentiment_price_eda`/`sentiment_decay`) are gone. |

Re-verified via repo-wide grep for each path string *after* the code moves above — zero remaining
hits outside `archive/`. All three were already gitignored (`data/sentiment_*/` wildcard in
`.gitignore`, unchanged — the wildcard already covers the new `archive/data/sentiment_*/` paths
from an earlier archival pass this session).

## Verification performed

- `python -m pytest --collect-only -q` before vs. after: 343 → 271 tests collected (expected drop,
  matching the ~12 archived baselines' own tests leaving collection scope), **9 collection errors
  both before and after — identical set, none new** (all pre-existing: archived-module imports in
  root `tests/`, missing optional `torch_geometric`/`mlflow`). Archiving
  `alignment_loss_baseline`/`gated_crossattn_baseline` removed the 2 module-name-collision errors
  those specifically caused (per the P2.3 fix's own finding) — net effect on the error count is
  zero because 2 errors left and this pass didn't fix the other pre-existing ones, not because
  nothing changed.
