# Summary — Dual-Group Retrain (12 New Sources) + SOTA Research + Directed-Spillover/QLIKE Baseline (2026-07-26)

User authorized this session to run fully unattended: "Tôi đồng ý, hãy làm hết toàn bộ, không cần
chờ tôi approve... hãy nghĩ ra các phương pháp tối ưu khác bằng cách deep research SOTA và đưa ra
kế hoạch cải thiện, implement kế hoạch đó như baseline mới luôn... Giờ tôi đi đây." Everything
below ran without further check-ins, per that instruction. Also completed earlier this session
(before the user left): 2 embedding documentation files
(`docs/EMBEDDING_STORAGE_SPECIFICATION.md`, `docs/EMBEDDING_USAGE_IMPLEMENTATION_GUIDE.md`) and
cleanup of 2 old cache backups (~8.8GB freed, user-confirmed).

## Part 1 — Dual-group panel rebuild + retrain (12 new sources)

**Context:** the dual-group news panel (`data/features/dual_group_news_panel.parquet`) was last
built before this session's GPU cache expansion added 12 newly-classified mainstream press
sources (baophapluat, bnews, cand, dantri, giaoducthoidai, hanoimoi, plo, sggp,
tapchicongthuong, tienphong, viettimes, vov — see previous report,
`docs/reports/2026-07-26_0230_summaryOfUpdate_report.md`).

**Rebuild:** ran `build_dual_group_panel.py` (no code changes — it auto-discovers sources via the
already-updated `KHACH_QUAN_SOURCES` set). Took 1377.9s (~23 min). Result: **146,700 rows, 148
cols, 30 tickers × 4890 dates, 81.49% coverage, 0 cache misses (0 PhoBERT calls)** — every article
was already in the expanded cache from the earlier GPU encode run.

**Retrain:** `train_dual_news.py --epochs 10` (unchanged code, same panel path).

| Metric | Old (pre-12-source, 2026-07-25) | New (12-source rebuild, 2026-07-26) | Diff |
|---|---|---|---|
| Test DirAcc | 68.71% | 68.25% | -0.46pp |
| Test R² | 0.7148 | 0.7124 | -0.0024 |
| Test QLIKE | 0.5458 | 0.5598 | +0.0140 (worse) |
| Test RMSE | 0.002640 | 0.002651 | +0.000011 (worse) |

**Verdict:** the 12 new sources did **not** improve the panel — a small, consistent *regression*
across all 4 metrics, not an improvement. Consistent with this project's repeated finding that
adding more news volume (mostly general-press, not finance-focused) tends to dilute signal rather
than add it.

## Part 2 — SOTA research (web search, 2025–2026 literature)

Given the long, consistent string of null results across every previous news-fusion variant
(dual-group, macro, gated cross-attn, selective/top3 gate, ablation, REST-TS, alignment loss),
searched current literature for a genuinely different angle rather than another news-branch
variant. Key finding: **every baseline in this project has kept 2 components fixed across all
prior experiments** — (a) the inter-stock graph (`graph_correlation.py`: always symmetric,
same-day Pearson correlation, either threshold or k-NN), and (b) the training loss (always plain
MSE; QLIKE only ever used as an eval metric).

Two 2025–2026 papers directly address both:
- **Zhang, Pu, Cucuringu & Dong (2025)**, *"Forecasting realized volatility with spillover
  effects: perspectives from graph neural networks"*, IJF 41(1) — nonlinear GNN spillover graphs
  beat linear GHAR; **training with QLIKE loss substantially outperforms plain MSE**.
- **Chi et al. (2026)**, *"Global Stock Market Volatility Forecasting Incorporating Dynamic
  Graphs and All Trading Days"*, J. Forecasting — directed volatility-spillover-index graphs
  (Diebold-Yilmaz-style) beat correlation-based graphs across all 8 tested markets.

Both converge on an LSTM+GNN architecture the project's `src/lstm_gat_hybrid` already implements —
confirming the base architecture is sound; the graph construction and loss function were the
identified gaps.

## Part 3 — New baseline: `2026-07-26_spillover_qlike_baseline`

**Two literature-grounded changes, both scoped to avoid the project's known Softplus-collapse
failure mode** (CLAUDE.md's documented 2026-06-21 LSTM-GNN normalization incident):

1. **Directed lead-lag volatility-spillover graph** (`graph_spillover.py`,
   `construct_directed_spillover_graph`): replaces the symmetric same-day correlation graph with
   an asymmetric one — edge `i←j` weighted by `corr(vol_j[t], vol_i[t+1])`, top-k incoming edges
   per receiver, same O(n²) cost as the existing graph, same per-window construction (no new
   leakage surface). Verified to match the existing `GraphAttentionLayer`'s masking convention
   (traced the attention-score/softmax dims to confirm row=receiver, col=transmitter) — no model
   code changed.
2. **QLIKE-augmented loss** (`losses.py`, `combined_loss`): `MSE_norm + 0.1 × QLIKE(denorm,
   clamped)`. Output layer stays linear/unchanged (no Softplus) — QLIKE only enters as a loss-side
   regularizer on the affine-inverse-transformed prediction, clamped to avoid NaN/Inf.

**Structure:** full 5-subfolder baseline (requirements/design/code/code_review/test), hard-isolated
(copy-modified sibling files per CLAUDE.md §3.F rule 3, no edits to any other baseline or to
`src/`). 20/20 pytest pass (graph asymmetry/top-k/degenerate-window properties, QLIKE-formula
correctness, combined-loss gradient finiteness, end-to-end train_epoch smoke). Self-adversarial
code review: 1 minor nit fixed (spurious noqa comment); verified no leakage, correct directionality
convention, no Softplus-style collapse risk (see `code_review/code_review_2026-07-26.md`).

### Real result (10 epochs, rebuilt panel, same data as Part 1's retrain)

| Metric | Dual-group (symmetric graph + MSE) | Spillover+QLIKE (directed + MSE+QLIKE) | Diff |
|---|---|---|---|
| Test DirAcc | 68.25% | 68.23% | -0.02pp |
| Test R² | 0.7124 | 0.7132 | +0.0008 |
| Test QLIKE | 0.5598 | 0.5622 | +0.0024 (worse) |
| Test RMSE | 0.002651 | 0.002647 | -0.000004 (negligible) |

**Verdict: NULL.** Directed spillover graph + QLIKE-augmented loss produced results
statistically indistinguishable from the symmetric-graph/MSE-only baseline on the same panel —
no meaningful lift on any of the 6 mandatory metrics. Neither beats HAR-only (69.98% DirAcc) or
the gated-crossattn record (R²=0.7157, QLIKE=0.557).

## Cross-session pattern (now ~10 independent null results)

dual-group, macro news, gated cross-attn, selective gate, top3 gate, ablation-derived gate,
REST-TS, alignment loss, pure-market, market-fallback, latent-noise, 12-source dual-group rebuild,
and now directed-spillover+QLIKE — **every architectural and feature-engineering variant tried in
this project has failed to clearly beat the HAR-only baseline on directional accuracy.** This is
no longer "one baseline didn't work" — it's a structural finding: with this dataset (VN30 daily
OHLCV + Vietnamese financial news), incremental changes to the news branch, the inter-stock graph,
or the loss function do not move the needle. The bottleneck is more likely upstream (feature
granularity — daily bars vs. intraday, target definition — 5-day-ahead Parkinson vol, or the
underlying premise that near-term VN30 volatility is largely unpredictable beyond HAR's own
autocorrelation structure) than in any of the architectural knobs turned so far.

## Files

- `data/features/dual_group_news_panel.parquet` — rebuilt (12 new sources, 146,700 rows).
- `results/dual_group_news_2026-07-26_192414/` — retrained dual-group baseline, results.json.
- `baselines/2026-07-26_spillover_qlike_baseline/` — new baseline, full 5-subfolder structure.
- `results/spillover_qlike_2026-07-26_192749/` — real 10-epoch run, results.json + learning curves.
- `results/spillover_qlike_2026-07-26_191055/` — smoke run (2 epochs, dummy news), kept for
  reference.
- `models/dual_group_news_2026-07-26_192414/best.pt`, `models/spillover_qlike_2026-07-26_192749/best.pt`
  — checkpoints.
- `docs/EMBEDDING_STORAGE_SPECIFICATION.md`, `docs/EMBEDDING_USAGE_IMPLEMENTATION_GUIDE.md` —
  written earlier this session (cache structure, cross-project reuse, train/val/test split
  handling for the news-embedding cache).

## Tests + code review

- `pytest baselines/2026-07-25_dual_group_news_embedding_baseline/test/
  baselines/2026-07-26_spillover_qlike_baseline/test/` → **26/26 pass** (no regressions from the
  panel rebuild or the new baseline).
- Self-adversarial code review on the new baseline (no `/code-review` interactive checkpoint —
  user explicitly authorized unattended execution): see
  `baselines/2026-07-26_spillover_qlike_baseline/code_review/code_review_2026-07-26.md`. 1 minor
  fix applied (spurious noqa); verified graph directionality convention against the GAT layer's
  actual masking semantics (the one bug class that would NOT show up as a shape/crash failure);
  verified no data leakage (graph built per-window, same as the pre-existing method); verified no
  Softplus-style prediction collapse risk (output layer unchanged, QLIKE is loss-side only).
- "Similar check" (CLAUDE.md DoD): grepped all baselines for the symmetric-graph pattern — **every
  one of the ~15 existing baselines** still uses it (list in code search, not reproduced here).
  Not retrofitted this session (out of scope — the comparison baseline already answered whether
  the graph/loss change helps: it doesn't, so retrofitting the other null-result baselines with
  the same non-improving change isn't warranted).
- Diff-coverage: **Not run** (tooling gap, already documented project-wide in CLAUDE.md).

## Risks / follow-ups for you to review

1. **The dual-group 12-source rebuild is a slight regression, not an improvement** — worth
   deciding whether to keep the expanded panel (more coverage, marginally worse metrics) or revert
   to the pre-expansion panel/cache for this specific baseline family.
2. **The spillover+QLIKE result is null** — the specific proxy tried (lag-1 lead-lag correlation,
   λ=0.1 untuned) doesn't rule out ALL directed-graph/QLIKE-loss approaches, just this exact
   implementation. Untried variants: proper Diebold-Yilmaz variance-decomposition graph (heavier
   to implement — VAR forecast-error decomposition), tuning `qlike_weight` beyond the single
   untuned 0.1 tried, or QLIKE as the ONLY loss (not additive with MSE).
3. **Structural finding (see Part 3 pattern section):** given ~10 independent null results across
   news features, graph structure, and loss function, recommend the next session's time is better
   spent questioning the upstream premise (is 5-day-ahead daily volatility on VN30 predictable
   beyond HAR at all, with any of the currently-tried feature families?) rather than trying an
   11th architectural variant. Concretely: consider (a) an honest ceiling-check — how much does
   the best possible model (e.g., an oracle with access to future HAR features) improve over
   HAR-only, to see if there's headroom at all; or (b) intraday/higher-frequency data if
   available, since daily-bar granularity may be the actual limiting factor, not the model.
4. Both new result directories (`results/dual_group_news_2026-07-26_192414/`,
   `results/spillover_qlike_2026-07-26_19*/`) plus 1 old smoke run are on disk — no cleanup done,
   all kept per project convention (results/ always retained).

## DoD checklist

- [x] Code satisfies the request (rebuild+retrain done; SOTA research done; new baseline
      implemented + trained for real)
- [x] Tests written + run (26/26 pass, no regressions)
- [ ] diff-cover C0/C1 — Not run (documented tooling gap)
- [x] Adversarial self-review — findings fixed, documented in baseline's code_review/
- [x] Real-data smoke validated (smoke run before the real 10-epoch run)
- [x] Impact analysis — similar-check grep across all baselines for the graph pattern; no sibling
      files modified (hard isolation maintained)
- [x] Summary report — this file
- [x] Training policy — capped at 10 epochs, enforced in code (`MAX_EPOCHS` check raises if
      exceeded)
