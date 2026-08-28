# Summary of update — paper restructured to masked-panel VN30/VN100 three-model study

Date: 2026-08-22

## Final state

The SOICT paper now reports one study: HAR / LSTM / LSTM+GAT on the masked union-of-dates panel, VN30 and
VN100 only, five node features, all five metrics, date-clustered DM. Data are the clean 1e-2 positivity-floor
results. Presentation/restructure only; every number quoted from the source reports. No code, no data, no new
references.

Title: "Multi-Horizon Stock Volatility Forecasting: HAR, LSTM and Graph-Attention Models on Vietnamese
Equities" (running head "Multi-Horizon Volatility Forecasting").

Key applied constraints (per coordinator, consolidated): single linear HAR baseline on the five features (no
"HAR-X" term, no 3-lag HAR in tables); three models HAR/LSTM/LSTM+GAT (directed volume→PK Top-5 weighted
2-hop); primary horizons h1/h5, extended h10/h22; all five metrics with equal weight; objective wording (no
"fairest"/"rich"/"hard to beat"/"dominates"/"parsimonious"/"genuine"); date-clustered DM (two contrasts, LSTM
vs HAR and LSTM+GAT vs LSTM, on QLIKE and MAE); positivity floor stated in one Method sentence (1e-2 × per-node
train mean); S&P 500 / U.S. / cross-market content and the graphical-lasso robustness note fully removed.

## Files

| Path | Purpose |
|---|---|
| `docs/paper/soict_paper_complete.md` | Markdown draft (DOCX source) — full rewrite |
| `docs/paper/soict_harlstmgat.tex` | LaTeX (≤12 pp) — full rewrite; preamble, author block, figure `\includegraphics{diagrams/soict_harlstmgat.png}` preserved |
| `docs/paper/soict_harlstmgat.pdf` | Recompiled (pdflatex ×2), 8 pages |
| `docs/paper/soict_paper_complete.docx` | Re-exported via pandoc |

## Sources of truth (numbers quoted exactly)

- `docs/reports/2026-08-22_masked_rich_floor1e2_clean.md` — Tables 1–3 (VN30/VN100 metrics + DM contrasts,
  1e-2 floor). The paper's single "HAR" row = the report's HAR-X (5-feature linear); the LSTM+GAT row = the
  report's LSTM+wGAT(vol→PK).
- `docs/reports/2026-08-22_lstm_qlike_blowup_diagnosis.md` — basis for the 1e-2 floor choice (cited only via
  the one Method floor sentence).

## Honest results (verified against the 1e-2 clean report)

- HAR has the lowest QLIKE on both panels at every horizon; no learned model has a significantly lower QLIKE
  than HAR (date-clustered DM, p≥0.05).
- Short horizons (h1/h5): deep models have the lowest MAE on VN100 (LSTM vs HAR p<0.001); LSTM+GAT has a
  significantly lower QLIKE than the no-graph LSTM in both panels (VN100 h1 p<0.001, h5 p=0.022; VN30 h1
  p<0.001, h5 p=0.031), reaching HAR's QLIKE level at VN100 h5 (0.5690 vs 0.5633) — helps the deep model, does
  not surpass HAR.
- Extended horizons (h10/h22): HAR lowest MSE/RMSE/QLIKE/R²; LSTM+GAT-vs-LSTM QLIKE not significant (p≥0.55).

## Verification / code review

Self adversarial review (number-accuracy focus):
- Re-derived and re-checked all Table 1/2 metric values and all Table 3 DM p-values + favoured-model letters
  against the 1e-2 source report, cell by cell (LSTM-vs-HAR = source LSTM-vs-HAR-X; LSTM+GAT-vs-LSTM = source
  wGAT-vs-LSTM).
- Confirmed the abstract headline numbers match the source: VN100 h5 QLIKE HAR 0.5633 vs LSTM+GAT 0.5690;
  wGAT-vs-LSTM QLIKE p (VN100 h5)=0.022, (h1)<0.001; VN100 h1 MAE LSTM 2.821 vs HAR 2.898.
- Scans: 0 occurrences of "S&P"/"sp500"/"U.S."/"cross-market"/"457", 0 "graphical-lasso"/"glasso",
  0 "HAR-X"/"HARX", 0 promotional/editorial adjectives (fairest/rich/hard-to-beat/dominates/parsimonious/
  genuine) in either file (the string "HARd to beat" remaining is the literal Audrino reference title).
- LaTeX: all `\ref` targets resolve; 0 undefined references/citations; no dangling `tab:sp500`/`sec:sp500`.

## Build evidence

- `pdflatex ×2` → `Output written on soict_harlstmgat.pdf (8 pages ...)`; 0 undefined references. ≤12 pp.
- `pandoc docs/paper/soict_paper_complete.md -o docs/paper/soict_paper_complete.docx --resource-path=docs/paper`
  → exit 0.

## DoD checklist

- Code: N/A (docs/presentation only); no code added → no tests required.
- Data-quality gate (Pandera/Evidently): N/A (no data change).
- Coverage/diff-cover: N/A (no code change).
- Code review: self adversarial number-accuracy review done; findings fixed.
- Performance/batching: N/A (no train/inference/data-processing code).
- Build: PDF (8 pp, 0 undefined refs) + DOCX regenerated.
- Objective wording enforced throughout; results stated factually per horizon on all five metrics.

## Risks / follow-ups

- Architecture figure (`diagrams/soict_harlstmgat.png`) retained; caption describes the directed
  volume→Parkinson weighted graph. The diagram image itself was not regenerated.
- At 8 pages there is ample room under the 12-page limit for figures/expansion if desired.
