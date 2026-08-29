# Summary of update — sector-graph ablation (HNX-primary)

**Date:** 2026-08-29
**Scope:** New baseline `baselines/2026-08-29_sector_gat_ablation/` (CLAUDE.md §3.F). CPU-prep + wiring +
a CPU directional training check on HNX. No GPU training (the overnight run owns the RTX 4060).

## What changed

A panel-agnostic sector-graph ablation for the LSTM+GAT volatility model. It swaps a **static
sector-defined edge** for the shipped statistical edge (directed volume-shock→Parkinson) under the exact
read-only `MaskedRichNet` / HAR-X pipeline (no live-file edits — the adjacency is passed as a parameter).

### Files (path → purpose)

| Path | Purpose |
|---|---|
| `code/sector_adjacency.py` | `build_sector_adjacency(tickers, sector_map, top_k)` → `[N,N]` float32 (self-loop, block-diagonal within sector); `load_sector_map`, `coverage`. Panel-agnostic. |
| `code/fetch_vn_sectors.py` | vnstock ICB (`Listing().symbols_by_industries()`, one bulk call) → canonical `ticker,sector` CSV. Lazy vnstock import (gate venv has none). |
| `code/fetch_sectors.py` | datahub GICS → canonical CSV (S&P500, retained but deprioritized). |
| `code/run_sector_ablation.py` | Build panel (reuse `_write_estimator_processed`+`build_masked_rich`), build `A_sector` aligned to `D.tickers`, dry forward-pass smoke, `--train-epochs N` CPU comparison. **CPU forced** (`CUDA_VISIBLE_DEVICES=""` before torch import) to protect the live GPU job. |
| `vn_sectors.csv`, `vn_icb_sectors.csv` | VN ICB sector map (canonical + raw dump) with provenance (source + fixed date `2026-08-29`). |
| `sp500_gics_sectors.csv` | S&P500 GICS map (provenance). |
| `test/test_sector_adjacency.py`, `test/test_runner_and_fetch.py`, `test/test_smoke_forward.py` | TDD property tests + fetch/runner coverage + real-data-sample CPU smoke. |
| `requirements/`, `design/`, `code_review/` | §3.F artifacts. |

## Sector-label coverage

- **HNX (primary):** 153/154 built-panel tickers mapped to ICB sectors (**99.35%**), 23 sectors,
  avg off-degree 10.8, 4 singletons. Screened-universe pre-build coverage 160/162 (98.8%). Unmapped
  (CAR, NST) → singleton own-sector (self-loop only). Source: vnstock ICB, one bulk call.
- **VN100 (secondary):** map available (same vnstock ICB CSV covers 697 symbols); run deferred.

## HNX directional result (Parkinson, h1)

**DIRECTIONAL CHECK ONLY** — CPU, 5 epochs, 1 seed (42), N=154, 60,028 test obs. Not a final number
(scale-up = 5 seeds, both horizons, on GPU once free).

| model | MSE | RMSE | MAE | QLIKE | R² |
|---|---|---|---|---|---|
| **sector_GAT** | 1.0e-6 | **0.001194** | **0.000648** | **1.8921** | **0.2019** |
| stat_GAT_vol2pk | 1.0e-6 | 0.001200 | 0.000675 | 1.9164 | 0.1948 |
| no_graph_LSTM | 1.0e-6 | 0.001213 | 0.000717 | 1.9153 | 0.1770 |

Date-clustered Diebold–Mariano on QLIKE (favours lower-loss model):

| comparison | mean_diff | p-value | favours |
|---|---|---|---|
| sector-GAT vs stat-GAT | −0.0232 | **0.0069** | sector-GAT |
| sector-GAT vs no-graph LSTM | −0.0236 | **0.0101** | sector-GAT |
| stat-GAT vs no-graph LSTM | −0.0004 | 0.969 | tie |

**Reading:** at this directional setting the **sector edge beats both the statistical edge and the
no-graph LSTM on all five metrics**, and the QLIKE improvement is statistically significant under
date-clustered DM (p<0.01 for both). The **statistical edge is indistinguishable from no-graph**
(p=0.97) — consistent with prior findings that the frozen statistical graph adds no OOS value, and it
localises the benefit to the *static, leakage-free sector structure*. Single-seed / 5-epoch caveat:
treat as a green light for scale-up, not a publishable number.

Result artifact: `results/sector_gat_ablation/sector_ablation_hnx_h1.json`.

## Tests + coverage

- 34 tests pass under the GPU venv (`.venv_gpu_encode`). TDD: sector-adjacency property tests written
  RED first, then implemented GREEN.
- Diff-coverage on changed lines: **C0 line 100%**, **C1 branch 98%** (gate: 100 / 95).
- ruff `--select F` (blocking set): clean. `E702` semicolons match house style (WARN-only).

## Code review

Self adversarial 3-lens review (`code_review/code_review_2026-08-29.md`). One MAJOR found and fixed:
the CPU-force guard used `setdefault`, which could leak the GPU if `CUDA_VISIBLE_DEVICES` was preset —
changed to a hard assignment. No open critical/major.

## Performance

No new batch=1 anti-pattern: training reuses the delivered batched GAT (`batch_size=512`, block-mask
adjacency, single-tensor forward). CPU is forced only to avoid contending with the live GPU job; the
2-hop GAT over N=154 is CPU-slow (a full run took ≫30 min under contention), so the directional check
used 1 seed / 5 epochs. Scale-up flips `SECTOR_ABLATION_FORCE_CPU=0` for GPU.

## Data-quality gate

**N/A (no data change)** — only new metadata CSVs inside the baseline folder; no `data/` file touched.
Pre-push gate ran the raw+processed data-quality + lessons-regression suites anyway (329 passed) plus
delivered-baseline tests (69 passed).

## Ready-to-run scale-up (after GPU frees)

```bash
for h in 1 5; do
  SECTOR_ABLATION_FORCE_CPU=0 .venv_gpu_encode/Scripts/python.exe \
    baselines/2026-08-29_sector_gat_ablation/code/run_sector_ablation.py \
    --panel hnx --horizon $h --train-epochs 10
done
# VN100 (secondary): same, --panel vn100
```

## DoD checklist

- [x] §3.F folder + 5 subfolders (requirements/design/code/code_review/test)
- [x] TDD (failing test first for the builder), surgical, no live-file edits
- [x] Tests pass + diff-coverage C0 100% / C1 98% on changed lines
- [x] Code review run + MAJOR fixed
- [x] HNX directional result produced (CPU) + DM
- [x] Committed (`5cf2b0e`) — on `origin/master` (concurrent session integrated it; local in sync)
- [ ] `results/.../sector_ablation_hnx_h1.json` produced AFTER the code commit — to be committed by the
      next push (see risks)

## Risks / follow-ups

- The push of `5cf2b0e` was initially rejected (remote advanced mid-gate); a concurrent session then
  integrated the commit, so it is on `origin/master` and local HEAD is in sync (0/0). Per CLAUDE.md,
  no force-push / no autonomous rebase was performed.
- The HNX result JSON was written after the code commit; it is a generated artifact and can be
  committed on the next push or regenerated via the command above.
- Numbers are single-seed / 5-epoch directional — confirm with 5 seeds × {h1,h5} on GPU before any
  paper claim. VN100 secondary run still pending.
