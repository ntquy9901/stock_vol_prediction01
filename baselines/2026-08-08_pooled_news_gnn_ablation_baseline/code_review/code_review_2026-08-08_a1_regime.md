# Code review — A1 common-date regime (T1.1)

Scope: commit a14482d and follow-up fixes on branch `feature/pooled-news-gnn-pilot`.
Changed files (all under `baselines/2026-08-08_pooled_news_gnn_ablation_baseline/`):
`code/data.py`, `code/run_pilot.py`, `test/test_regime.py`, `test/test_train_smoke.py`.

Method: adversarial review (cynical bug hunter) focused on data leakage and correctness,
plus the standard acceptance check against the A1 requirement (only the training-sample SET
may differ between pooled and common-date regimes).

## Result

No CRITICAL or HIGH findings. The leakage guard central to A1 holds:

1. Scaler/winsor reuse (no refit): `build_screening_inputs` fits `store` on train frames,
   builds the pooled manifest, optionally attaches news, then calls
   `restrict_manifest_to_common_dates`, which only subsets sample tuples and passes
   `preprocessing_hash`/`ticker_to_id`/`exclusions` through unchanged. `common_trading_dates`
   transforms with the already-fit store (`transform_frame` never refits). Verified by
   `test_common_date_regime_reuses_scalers_and_reduces_samples` on the real (non-mocked) path
   (`store.to_dict()` identical; common-date train ⊂ pooled train).
2. Common-date axis: intersection of per-ticker post-HAR trading dates; restriction requires
   both every `input_date` and the `target_date` to lie on the axis. Restriction can only
   remove samples, so common-date ⊆ pooled is structural.
3. Refactor safety: `build_graph_manifest` behavior is preserved by `_transform_full_frames`
   (same validation order, same ticker-id derivation, `set_index("date")` reapplied
   identically). Full existing graph tests remain green.
4. News-panel identity: news is attached with full-train cutoffs before restriction, so the
   loaded panel/provenance and every surviving sample's news vectors are identical across
   regimes.

## Findings and resolutions

- LOW-MEDIUM — `--regime` silently ignored for `--phase graph`. Resolved: `parse_args` now
  rejects `--phase graph --regime common-date` (`test_parse_args_rejects_common_date_regime_for_graph_phase`).
- LOW — common axis uses full-history HAR warm-up while the pooled manifest uses per-split
  warm-up (benign for leakage; restriction is a subset). Resolved by documentation: the
  `restrict_manifest_to_common_dates` docstring states windows stay per-ticker contiguous and
  the horizon gap is counted in per-ticker days, not common-axis steps.
- LOW — no integration test on the P2/P3 attach-then-restrict news path. Resolved:
  `test_common_date_news_path_is_byte_identical_to_pooled_per_key` asserts per-key news
  tensor/mask equality across regimes and preserved news width.

## Checks

- `pytest baselines/2026-08-08_pooled_news_gnn_ablation_baseline/test/ -q`: 125 passed.
- `ruff check code/ test/`: clean.
