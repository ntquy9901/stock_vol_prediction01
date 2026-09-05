# Paper v3 changelog and open TODOs

- **File:** `docs/paper/soict_harlstmgat_2026-09-05_v3.tex` (compiled to
  `docs/paper/soict_harlstmgat_2026-09-05_v3.pdf`, 15 pages, pdflatex x2, MiKTeX).
- **Base:** `docs/paper/soict_harlstmgat_2026-09-04_v2.tex` (v2 structure, prose, and all delivered
  tables/numbers reused verbatim; v3 only ADDS material). Primary/headline market remains VN100.
- **Layout rules followed:** `docs/paper/PAPER_LAYOUT_RULES.md` (strongest market first; Table 1 =
  HAR-X vs VolGA; Table 2 adds LSTM for graph ablation; progressive tables; objective voice; honest,
  DM-backed, all horizons {1,5,10,22}; no DirAcc; Parkinson variance sigma^2; scale in caption not
  e-notation; best value per column bold).

## What is new in v3 vs v2

1. **New ablation subsection (Sec. Edge construction, `\label{sec:edge}`)** placed under
   "Ablation studies", after "Widening the training universe" and before "Alternative volatility
   estimators". Reports the contemporaneous-correlation edge probe on VN100.
2. **New Table `tab:edgemetrics`** — contemp-edge VN100 metrics: HAR / HAR-X / LSTM / VolGA x
   h{1,5,10,22} on MSE / RMSE / MAE / QLIKE.
3. **New Table `tab:dm-edge`** — contemp-edge date-clustered DM contrasts: VolGA-vs-LSTM (graph) and
   VolGA-vs-HAR-X, each on QLIKE / SE / AE, all four horizons (wrapped in `\resizebox` to fit width).
4. **Abstract** — one sentence added noting the contemporaneous-edge probe does not overturn the
   reading (does not improve the graph's QLIKE margin, significantly hurts at h22).
5. **Related Work / Experiments** — one clause each pointing to the edge-construction ablation.
6. **Discussion** — new "Edge construction" paragraph with the honest verdict + project edge-EDA
   context (vol->PK edges weak ~1.34x above shuffled null, unstable ~5% train->test Top-5 overlap;
   contemp edge more stable ~25% but largely redundant with the market_pk node feature).
7. **Limitations** — new "Edge-probe protocol" caveat (shorter walk-forward, within-run only).
8. **Conclusion** — one sentence added on the edge probe.

No delivered v2 table or number was changed.

## Honest treatment of the contemporaneous edge (stated explicitly in Sec. edge)

- **Different, shorter protocol -> cross-run comparison confounded.** Contemp probe folds/seeds vary
  by horizon: h1/h5 = 7 folds, 3 seeds; h10/h22 = 6 folds, 5 seeds (from the JSON `n_folds`/`seeds`),
  vs the delivered main run's 22 folds, 5 seeds. Only within-run comparisons are valid; a contemp cell
  is NOT level-comparable to a delivered vol->PK cell (caption states this).
- **Confound demonstrated at h1:** HAR (edge- and seed-independent) essentially unchanged across runs
  (QLIKE 0.4985 probe vs 0.4983 delivered), but the no-graph LSTM ensemble QLIKE is markedly lower in
  the short probe (0.4938 vs 0.5025 delivered) even though the LSTM uses no edge -> the lower probe
  VolGA QLIKE (0.4889) is largely a protocol effect, not the edge.
- **Graph marginal value (VolGA vs no-graph LSTM):** contemp edge is NOT an improvement over vol->PK
  on QLIKE — loses the short-horizon significance (h1 p=0.223, h5 p=0.054) vs delivered vol->PK
  (h1 p=0.008, h5 p=0.011), and the graph SIGNIFICANTLY HURTS at h22 (p=0.001 favouring the no-graph
  LSTM). On SE/AE the contemp graph is significant only at h1 (SE p=0.042, AE p=0.002, favour VolGA) —
  a loss-dependent short-horizon point-error gain only.
- **Net verdict:** consistent with the paper's overall finding — the graph's marginal value is loss-
  and horizon-dependent and does not robustly beat the no-graph LSTM; changing the edge does not
  overturn this. A matched-protocol vol->PK run is required to isolate the edge effect.

## Provenance (every number from a stored JSON)

- Delivered main + graph + VN30 tables + pooled/transfer ablation: unchanged, carried from v2
  (`results/walkforward_volga/walkforward_volga_{vn100,vn30}_h{1,5,10,22}.json`,
  `results/pooled_transfer_vn30/pooled_vn30_h{1,5,10,22}.json`).
- Contemp-edge tables: `results/contemp_edge/contemp_contemp_vn100_h{1,5,10,22}.json`
  (`metrics[model].{mse,rmse,mae,qlike}`; `dm_date_clustered.{VolGA_vs_LSTM,VolGA_vs_HARX}.{qlike,se,ae}`;
  model key `LSTM_wGAT_vol2pk` = VolGA under the contemp edge).
- Confound cross-check numbers (HAR/LSTM/VolGA h1) read from
  `results/walkforward_volga/walkforward_volga_vn100_h1.json` (delivered) vs
  `results/contemp_edge/contemp_contemp_vn100_h1.json` (probe).

## Compile status

- pdflatex x2, no undefined references or citations. 15 pages total (includes bibliography).
- Overfull hboxes: the new `tab:dm-edge` overfull was fixed with `\resizebox`. Three remaining
  overfull hboxes (VN100 graph table, VN30 table, pooled table) are pre-existing v2 tables and were
  left untouched (surgical-change policy).
- Aux files removed; only `.tex` + `.pdf` kept.

## Open TODOs

- **VN30 contemporaneous-edge run** is still in progress; not reported (no VN30-contemp numbers
  invented). Add when the JSON lands.
- **Matched-protocol vol->PK run** (same folds/seeds as the contemp probe) needed to isolate the pure
  edge effect from the protocol effect — listed as future work in Sec. edge and Limitations.
- **Alternative-estimator tables** (Rogers-Satchell, windowed Yang-Zhang across markets): still
  `[provisional]`, pending clean-data walk-forward regeneration.
- **Additional markets** (HNX, HOSE, S&P 500): still `[provisional]`, pending clean-data walk-forward.
- **Page budget:** 15 pages total including references vs the 12-pages-excluding-references SOICT
  limit — trimming likely needed before submission (not attempted here; adding-only scope).
- **Contemp QLIKE dispersion:** probe JSONs store only ensemble `metrics` (no `metrics_per_seed`), so
  `tab:edgemetrics` reports QLIKE point values without a +/- per-seed std (noted in the caption).
- **Protocol note:** the contemp probe's fold/seed counts differ by horizon (h1/h5 = 7f/3s;
  h10/h22 = 6f/5s); this is disclosed in the paper, but a uniform re-run would make the probe cleaner.
