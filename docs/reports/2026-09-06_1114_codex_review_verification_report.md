# Verification of Codex `edge_hmatched` reviews — impact on current tasks + fixes

Independent re-review of three Codex reports (`C:\tools\codex_ml_senior_architect_agent\reports\edge_hmatched_review_{sp500,vn100,vn30}.md`),
verifying each Codex finding against the actual driver code, the 12 result JSONs, and the new paper draft
(`docs/paper/volga_new_2026-09-06_skeleton.tex`). Codex's numeric readings all match the JSONs; the disagreement
is over which findings still apply given the owner's decision to retire the old fixed-lag VolGA comparison.

## Verdict summary

| Codex finding | Verified | Classification | Action |
|---|---|---|---|
| Missing fixed-edge control → cannot claim horizon-matching beats fixed-lag (central, all 3) | Correct that no fixed-edge model is in the JSONs | **MOOT** — the paper makes no such claim (contributions = VolGA vs LSTM, VolGA vs HAR-X only) | None |
| No multiple-testing correction on paper DM p-values; SP500 h22 borderline | Correct: `_dm_all` returns raw p; SP500 h22 p=0.047 was called "significant" | **VALID-FIX** | SP500 h22 sentence reworded; added multiplicity note to Experiments |
| Edge target mislabeled as $\sigma^2$ (units trap) | Correct: code correlates volume z-score with `sqrt(pk)`=$\sigma$; figure said $\sigma^2_j$ | **VALID-FIX** | Figure label $\sigma^2_j\to\sigma_j$; edge subsection clarified (z-scored source, target=$\sigma$=sqrt Parkinson variance) |
| VN30 "33 tickers" inconsistent with 31 trained nodes | Correct (`num_nodes=31`) | **VALID-FIX** | Data section: "VN30 (33 tickers, 31 after screening)" |
| Provenance gaps in JSON (git commit, data fingerprint, fold dates, Top-K, QLIKE floor, per-fold density) | Correct: only seeds/n_folds/num_nodes/edge_sig_alpha/edge_density_mean present | **VALID-SHOULD-FIX** (reproducibility) | Follow-up: add fields to driver re-serialization (needs a re-run to populate; not blocking claims) |
| Stale 3-seed chain log vs 5-seed JSONs | Correct (cosmetic; JSONs internally correct) | **VALID-SHOULD-FIX** | Follow-up: annotate/remove stale `_tmp_*_chain.sh` logs |
| 7-fold not labeled exploratory vs final | Run is 7-fold; paper says "seven retrain points" (honest, unlabeled) | **VALID-SHOULD-FIX** (framing) | Optional one-word "final"; deferred (no false claim) |
| Floored/limit-lock days dominate QLIKE | Valid; being addressed by `qlike_robust` | **IN PROGRESS** (task #15) | VN re-run populating robust; robust table to be added |
| Sparse long-horizon edge (density ~0.47% h22) | Correct and by-design (Bonferroni fallback) | **ALREADY-DONE** | Already discussed in paper |
| Substantive result readings (VolGA>LSTM on SP500 h1/h10; HAR-X best; VN n.s.; VN30 graph worse on QLIKE) | All match JSON and paper | **ALREADY-DONE** | None |
| Future work: smaller Top-K, pooled cross-market, Transformer baseline, bootstrap CIs | Not done | **FUTURE WORK** | List in Limitations (existing) |

## No MUST-FIX
No surviving false claim invalidates a headline result. The only claim that failed verification is SP500 h22
significance without multiplicity correction; it is now corrected. SP500 h1 (p=7.7e-22) and h10 (p=6e-4) survive
a Bonferroni correction across the four horizons; VN30 h1 HAR-X-vs-VolGA (p=0.009) survives.

## Impact on current tasks
- **#15 (VN re-run for qlike_robust + paper update): reinforced.** The robust-QLIKE work directly addresses
  Codex's floored-days concern; finishing it and adding the table closes that finding.
- **#12/#13/#14: unaffected.** Regime NO-GO and the architecture figure stand; the figure needed only the
  $\sigma^2\to\sigma$ edge-label correction (applied).
- The central Codex objection (fixed-edge control) is MOOT under the owner's decision, so no new experiment is
  required.

## Fixes applied (paper, compiled 8 pp)
1. Figure edge label $\mathrm{corr}(\mathrm{vol}_i(t),\sigma^2_j)\to\sigma_j$ (units correctness).
2. Edge subsection: source = z-scored volume shock; target = $\sigma_j$ (sqrt Parkinson variance).
3. SP500 h22: reworded from "significantly lower" to lower at a nominal p=0.047 that does not survive Bonferroni.
4. Experiments: added a sentence reading borderline p-values under multiplicity.
5. Data: "VN30 (33 tickers, 31 after screening)".

## Deferred (SHOULD-FIX, non-blocking follow-ups)
- Add provenance fields (git commit, QLIKE floor, edge Top-K, per-fold edge density, fold date ranges) to the
  driver's result serialization; populate on the next re-run.
- Remove/annotate stale `_tmp_*_chain.sh` logs so the 3-seed command cannot be mistaken for the run of record.
- Optional: label the seven-fold walk-forward as the final protocol; add bootstrap CIs.
