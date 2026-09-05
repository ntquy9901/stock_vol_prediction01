# Code review — edge_horizon_matched (2026-09-05)

Scope: `baselines/2026-09-05_edge_horizon_matched/code/run_edge_hmatched.py` (181 lines) — the driver used
to train the horizon-matched VolGA edge experiment, including on `sp500_clean` via the Colab notebook.
Layers: correctness (Blind Hunter), edge cases (Edge Case Hunter), acceptance (Auditor), plus the repo's
config-hardcode and performance lenses.

## Findings

### MAJOR — config-hardcode: `lookback = 22` hardcoded (FIXED)
`run()` assigned `lookback = 22` as a bare literal (line 111). Violates the single-source-of-truth rule
and would be BLOCKED by the pre-push config-hardcode gate (identical pattern flagged in the sibling
`2026-09-04_contemp_edge`). Fixed by mirroring the delivered VolGA baseline: `run(..., lookback=pc.LOOKBACK)`
and a `--lookback` CLI argument with `default=22` (the experiment value that matches the delivered VolGA
edge; library default = canonical `pc.LOOKBACK`). No bare literal remains.
Verification: `pytest` 5/5 pass, `ruff --select F` clean, `grep` shows no `lookback = 22`.

### MINOR — performance: O(n²) Python double loop in `directed_vol2pk_hmatched`
Edge construction iterates every (target j, source i) pair with a per-pair NaN mask + `np.corrcoef`, on
CPU on the main thread. For `sp500_clean` (n≈497 nodes) this is ≈247k correlations per fold × folds. It is
NOT the GPU training hot loop — the edge matrix is built once per fold, while the LSTM/VolGA training (5
seeds × 3 models) dominates wall-clock on the A100. Left as-is: correctness (per-pair finite masking,
horizon-matched shift) is clearer in the explicit loop, and it is not the bottleneck. Vectorising the
masked cross-correlation is a possible future optimisation, not required for this run.

### INFO — `--market sp500` (non-clean) no longer resolves
After `data/processed_enriched/sp500` was archived (2026-09-05), `enriched_glob("sp500")` returns an empty
glob and the driver would fail on an empty universe. Intended: the canonical S&P dataset is `sp500_clean`,
which the notebook and bundle use. `enriched_glob("sp500_clean")` resolves correctly to
`data/processed_enriched/sp500_clean/*.csv` (verified). The stale `sp500` choice is harmless (fails loud on
empty glob) and left in the CLI `choices` list to avoid unrelated churn.

## Verified correct (no change)
- No look-ahead leakage: the train boundary `last_tr_row = last_train_anchor + horizon` includes exactly the
  realized targets available to training; the horizon-matched alignment `src=v[:-h]`, `tgt=p[h:]` correlates
  volume(t) with sqrt_pk(t+h) as documented.
- Bonferroni floor guarded for degenerate n (`max(n-1,1)`), zero-variance and `< min_pairs` pairs skipped.
- Result JSON keys (`metrics`, `edge_density_fix_mean`, `edge_density_hm_mean`, `dm_date_clustered`) match
  the fields the Colab notebook reads in its summary cell.
- No unused imports (`ruff --select F` clean). `run()`/`main()` are entry drivers marked `# pragma: no cover`;
  the tested surface is `directed_vol2pk_hmatched` + `_edge_density` (5 unit tests: h1 reproduces the
  delivered lag-1 edge, lead-lag detected only at the matching horizon, floor prunes noise vs no-floor,
  self-loop = 1, density in range).

## Status
MAJOR fixed and verified. MINOR/INFO documented, no action required for the SP500 run.
