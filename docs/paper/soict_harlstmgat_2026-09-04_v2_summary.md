# SOICT VolGA paper v2 — restructure summary (2026-09-04)

Restructures `soict_harlstmgat_2026-09-04.tex` (v1) to follow `PAPER_LAYOUT_RULES.md`
(strongest-result-leads ordering, minimal-then-additive table progression, objective voice).
No existing file was modified; three new files were created:

- `docs/paper/soict_harlstmgat_2026-09-04_v2.tex`
- `docs/paper/soict_harlstmgat_2026-09-04_v2.pdf` (pdflatex ×2, 12 pages)
- `docs/paper/soict_harlstmgat_2026-09-04_v2_summary.md` (this file)

## What changed vs v1

### 1. Headline market: VN30/VN100 (co-primary in v1) -> VN100 (single headline market)
- Abstract, Introduction, Data, Experiments and Results now lead with **VN100** (larger, more
  liquid; the panel where the graph branch is significant). VN30 is demoted to a cross-market
  ablation subsection.
- Honest rationale stated in-text: VolGA vs no-graph LSTM significant at VN100 h1 (p=0.008) and
  h5 (p=0.011); VolGA attains the lowest point QLIKE at VN100 h1 (0.4916).

### 2. Table progression (minimal -> +1 axis per table)
| # | Label | Shows | Models |
|---|-------|-------|--------|
| 1 | `tab:vn100main` | **Main result, VN100** — MSE/RMSE/MAE/QLIKE × h{1,5,10,22} | HAR-X, VolGA only |
| 2 | `tab:vn100graph` | **Graph ablation, VN100** — adds LSTM so VolGA−LSTM isolates the graph | HAR-X, LSTM, VolGA |
| 3 | `tab:dm-vn100` | **DM contrasts, VN100** — graph (VolGA vs LSTM, QLIKE) + VolGA vs HAR-X (QLIKE, MAE) | — |
| 4 | `tab:vn30` | **Market ablation, VN30** — metrics + DM-graph column (n.s. all h) | HAR-X, LSTM, VolGA |
| 5 | `tab:pool` | **Pooled/transfer VN30** — Arm1-vs-Arm0 DM on QLIKE/AE, all 4 horizons | LSTM, VolGA |

- v1 had 4 metric tables that each dumped all four models (HAR, HAR-X, LSTM, VolGA) plus R², with
  VN30 and VN100 as parallel co-equal tables and a combined DM table. v2 removes R² (rules §3
  columns = MSE/RMSE/MAE/QLIKE only), drops the plain HAR row from the tables (mentioned in prose
  only for VN30 QLIKE 0.4952), and builds up one variable at a time: Table 1 is the 2-model
  headline; Table 2 introduces LSTM; Table 4 introduces the VN30 market.
- Best value per column per horizon is bold; QLIKE for learned models is `value ± per-seed std`;
  scaling (MSE ×10⁻⁷, RMSE/MAE ×10⁻⁴) stated in captions, no e-notation in cells.

### 3. GARCH
- No GARCH results exist in these JSONs. v2 does **not** invent GARCH rows; GARCH is mentioned once
  in Related Work as a reference class "not run in the present walk-forward." (v1 also omitted GARCH
  rows; the HNX template's GARCH rows were not carried over.)

### 4. Honesty / voice (per rules §5)
- Headline is the graph branch's marginal value on VN100 short horizons + VolGA's lowest point QLIKE
  at VN100 h1; explicitly states no deep model significantly beats HAR-X on QLIKE at any
  market/horizon; effect is loss- and node-breadth-dependent (significant on VN100, null on VN30).
- No "hurts/never helps/fails"; all four horizons reported; no DirAcc; target stated as Parkinson
  variance σ²; data-quality limitations (unverified split adjustment, variance target, QLIKE floor)
  disclosed; HAR-X cited to Corsi 2009 + Clements et al. 2024.

### 5. Section order aligned to rules §4
Abstract (2 para) → Keywords → Introduction (with (i)/(ii)/(iii) contributions + date-clustered DM
rationale) → Terminology glossary → Related Work → Method → Fig 1 → Data → Table 1 + Experiments →
Results: VN100 → Ablations (7.1 graph/VN100, 7.2 VN30, 7.3 widening universe, 7.4 estimators
[provisional], 7.5 additional markets [provisional]) → Discussion → Limitations → Conclusion.

## Numbers — source of truth
All table numbers are read from the result JSONs and cross-checked against
`deliverables/2026-09-04_paper_pipeline_review/RESULTS_SUMMARY.md`:
- `results/walkforward_volga/walkforward_volga_vn100_h{1,5,10,22}.json` — VN100 `metrics[model].{mse,rmse,mae,qlike}`
  and `dm_date_clustered` (VolGA_vs_LSTM, VolGA_vs_HARX). VN100 h1/h5 metrics + DM p-values were
  read directly from the JSONs and match the table cells exactly (e.g. h1 VolGA qlike 0.49161…→0.4916;
  DM VolGA_vs_LSTM qlike p=0.008335…→0.008; DM VolGA_vs_HARX ae p=0.000224→<0.001).
- `results/walkforward_volga/walkforward_volga_vn30_h{1,5,10,22}.json` — VN30 metrics + graph DM
  (carried from v1's JSON-derived tables; QLIKE values confirmed against RESULTS_SUMMARY).
- `results/pooled_transfer_vn30/pooled_vn30_h{1,5,10,22}.json` — Table 5 (verbatim from v1 tab:pool,
  matches RESULTS_SUMMARY).

## Compilation
- `pdflatex` (MiKTeX, CPU) run twice; exit 0 both passes; **0 undefined references/citations**;
  **12 pages** total (including the 16-item bibliography). Figure `diagrams/soict_harlstmgat.png`
  present and embedded. Byproducts (.aux/.log/.out) deleted; only .tex + .pdf kept.

## Open TODOs / caveats
- **12-page limit:** the 12 pages *include* the reference list, so the body is within the SOICT
  "12 pages excluding references" limit — but this is at the edge; if the body alone must be checked,
  confirm the bibliography spillover page. No trimming was needed to compile.
- **Provisional sections (unchanged from v1):** §7.4 alternative estimators and §7.5 additional
  markets (HNX/HOSE/S&P 500) remain `[provisional]` with `% TODO` — no clean-data walk-forward runs
  yet. Estimator tables and cross-market tables are intentionally omitted, not fabricated.
- No numbers were fabricated; no placeholder cells were needed (every reported cell had a JSON value).
- Not run (out of scope per task): git operations, GPU/training, code-review, data-quality gate —
  this is a documentation-only restructure of an existing draft.
