# Adversarial code review — combo ladder (2026-08-14)

3-layer review (Blind Hunter + Edge Case Hunter + Acceptance Auditor), run before "done".

## Confirmed correct (all three layers)
- **Leakage clean:** train-only scalers (per-ticker `build_extended_store`, fixes prior H1), news
  cutoff on train `input_dates`, vol→PK edge frozen on each ticker's train split, graph-bound
  `allowed`, one-basis obs invariant asserted and held (val=14418, test=14464 == canonical basis).
- **P0 pure HAR:** `run_e0` slices the FIRST 3 columns (HAR) — correct under the 5-feature order;
  B's `run_har_reference` (last-3 slice) would have been wrong, deliberately avoided.
- **DM correct:** sign convention, seed-ensemble alignment, target/obs-set assertions, identical
  1e-8 QLIKE floor across each compared pair; verified by re-execution (numbers reproduce).
- **Edge wiring genuine:** `swap_adjacency` output is consumed by the graph rungs (provenance hash
  reflects the swapped adjacency) — no silent kNN fallback.
- **Positivity floor (H2) holds for the DM-tested rungs:** P0/P3/G1 all floored at 1e-6.

## Findings and resolutions

| # | Sev | Finding | Resolution |
|---|---|---|---|
| BH-1 | MEDIUM | P1/P2 (from B's `run_pooled_rung`) use QLIKE floor 1e-8, not the 1e-6 of P0/P3/G1 → P1/P2 **QLIKE** not like-for-like. Headline DM (P0/P3/G1) unaffected. | **Documented + mitigated (no re-run).** The report compares P1/P2 to HAR on floor-independent **RMSE/R²/MAE** (where they win), and explicitly flags P1/P2 QLIKE as not directly comparable. The node-features-beat-HAR-on-QLIKE result is already DM-established floor-consistently in the sibling `2026-08-11_eda_gnn_baseline` (E2 p=0.012). Consistent-floor P1/P2 QLIKE + their DM is the top follow-up (needs a floored news-P2 trainer / per-obs dump). |
| EC-X1 | MEDIUM | `aggregate` shared module name collides with eda_gnn's `aggregate.py` under repo-wide pytest. | **Fixed:** renamed to `combo_aggregate.py` (+ test). |
| EC-C1 | MEDIUM | `edge_count` counted diagonal self-loops → misleading `vol2pk_edges` and a trivially-true smoke assert. | **Fixed:** count off-diagonal non-zeros only (165 directed edges; the 198 total = 165 directed + 33 self-loops). |
| EC-A1 | MEDIUM | A degenerate (zero-variance) DM pair raised and aborted the whole summary. | **Fixed:** per-pair try/except records `{"error": ...}` and continues; covered by a new test. |
| EC-A2 | LOW | `favors` labeled an exact tie as "B". | **Fixed:** `mean_diff == 0` → "tie". |
| EC-A3 | LOW | Dead `DUMP_RUNGS` constant. | **Fixed:** removed (own orphan). |
| EC-A4 | LOW | `json.dumps` without `allow_nan=False`. | **Fixed:** `allow_nan=False`. |
| EC-C2 | LOW | Reused `_log` does not mkdir `ROOT/temp`; fresh checkout would `FileNotFoundError`. | **Fixed:** `stamp.parent.mkdir` in `main`. |
| EC-C3 | LOW | No in-code assertion of feature/news width. | **Fixed:** `build_basis` raises if `price_dim != 5` or `news_dim <= 0`. |
| EC-C4 | LOW | One-basis invariant passes on empty val/test (two empty sets equal). | **Fixed:** explicit non-empty guard per split. |
| EC-A5 | LOW | `_load_rows` silently collapses duplicate (ticker,date) rows. | **Accepted:** cross-seed set/target checks make an inconsistent duplicate detectable; upstream dumps do not duplicate. |
| BH-2 | LOW | `test_p0` asserts metrics exist but not the pure-HAR claim itself. | **Accepted:** run_e0's first-3 slice was statically confirmed by all three layers; a perturbation test is deferred (deepcopy of 73k samples is costly). |
| BH-3 | LOW (op) | Under Python 3.14 the mandated `code/` dir shadows stdlib `code`, breaking `pdb` import at pytest configure. | **Environment note:** tests run under `.venv_gpu_encode` (Python 3.10, the CLAUDE.md stack); shared by all baselines (dir name is §3.F-mandated), not combo-specific. |

## Post-fix verification
- `pytest baselines/2026-08-14_pooled_news_edanode_gnn/test/` → **5 passed** (`.venv_gpu_encode`, Py 3.10).
- `combo_aggregate.py` re-run: DM numbers reproduce identically (G1_vs_P0 QLIKE p=0.0035 favoring HAR).
- `ruff check` clean. No HIGH findings; all MEDIUM fixed except BH-1 (documented + mitigated as above).
