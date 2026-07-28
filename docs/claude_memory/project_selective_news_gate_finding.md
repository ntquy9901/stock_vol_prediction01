---
name: project-selective-news-gate-finding
description: 2026-07-25 finding — per-ticker news usefulness from a tree-model EDA does not transfer to the shared LSTM-GNN architecture
metadata:
  node_type: memory
  type: project
  originSessionId: 7b3b1f97-cfdd-4b28-b9f4-b53d0110952d
  modified: 2026-07-25T14:11:35.631Z
---

`baselines/2026-07-25_selective_news_gate_baseline/` tested masking the news branch on/off per
ticker, using a ticker list derived from `docs/suggestion/2026-07-25_professor_report.md` (HGB/
XGBoost per-ticker delta-R^2 at t+5). Result: **hypothesis contradicted**. The 22 "NEWS_ON"
tickers (EDA said news helps) averaged 46.29% test DirAcc; the 10 "NEWS_OFF" tickers (EDA said no
benefit / user-excluded SHB) averaged 51.60% — the opposite of what was predicted. Overall DirAcc
(67.56%) also came in below the unmasked dual-group baseline (68.50%) and HAR-only (69.98%).

**Why:** the EDA used a completely different model family (HGB/XGBoost, independent per-ticker
regression, ~500-col price+news_adv_full feature set) than this project's shared multi-stock
LSTM-GNN (32 stocks jointly, GAT-mixed HAR embeddings, 146-col dual-group+EWMA subset).
"Usefulness of news for ticker X" measured in one model family doesn't necessarily transfer to a
jointly-trained architecture with cross-stock attention.

**How to apply:** if asked to revisit selective/per-ticker news gating, don't reuse a different
model family's feature-importance signal again — derive the ON/OFF split from THIS
architecture's own per-ticker ablation (train with vs. without news per ticker, using the actual
LSTM-GNN) instead. Full writeup: `docs/reports/2026-07-25_1036_summaryOfUpdate_report.md`.

**Follow-up (`baselines/2026-07-25_top3_news_gate_baseline`):** tried narrowing to just the 3
tickers with the strongest 4-horizon avg delta-R^2 (VIB, ACB, MWG) instead of 22. Result:
inconclusive, not confirmatory — val showed ON>OFF (+5.6pp) but this did NOT replicate on test
(48.67% vs 48.89%, a tie). With only 3 tickers the group average is dominated by per-ticker noise
(8pp spread among just VIB/MWG/ACB). Neither the broad nor narrow EDA-derived selection produced
a convincing win — strengthens the conclusion that this transfer approach doesn't work reliably,
regardless of the evidence threshold. Writeup: `docs/reports/2026-07-25_1054_summaryOfUpdate_report.md`.

**3rd attempt (`baselines/2026-07-25_news_usefulness_ablation` + `..._ablation_derived_gate_baseline`):**
abandoned the HGB/XGBoost EDA entirely — derived a NEW 11-ticker ON list by directly comparing a
fresh HAR-only reference vs. the all-ON dual-group model, BOTH trained 10 epochs on the identical
data pipeline (epoch-matched, to avoid a training-budget confound caught and fixed mid-session —
see that baseline's code review), using per-ticker delta_QLIKE (continuous, far less noisy than
DirAcc). Result: NEWS_ON tickers (50.47% test DirAcc) beat NEWS_OFF (47.33%) by +3.1pp — the
right direction, unlike the two EDA-based attempts — but QLIKE itself (the metric the list was
selected on) barely moved vs. the HAR-only reference, and overall DirAcc (68.23%) still trails
both the all-ON model (68.50%) and HAR-only (69.98%). Best of 3 attempts, still inconclusive.
**Consolidated verdict:** after 3 different ticker-selection methods (broad EDA, narrow EDA,
internal ablation), none produced a baseline worth promoting over the plain all-ON dual-group
model. Next rigor step, if pursued further, is multi-seed ablation (this was single-seed
throughout). Writeup: `docs/reports/2026-07-25_1127_summaryOfUpdate_report.md`.

**4th check (2026-07-25, learned gate vs. all 3 external methods):** instead of another
externally-selected list, extracted the LEARNED per-ticker gate from the already-trained
`baselines/2026-07-18_gated_crossattn_baseline` checkpoint
(`models/gated_crossattn_2026-07-18_023500/best.pt`, test R²=0.716 — this project's best
news-fusion result) via a forward hook on `gate_mlp`
(`baselines/2026-07-18_gated_crossattn_baseline/code/analyze_gate_per_ticker.py`). Result: the
gate DID learn non-uniform per-ticker reliance (mean gate 0.0175 VHM to 0.091 BID, ~5x spread;
bank tickers BID/ACB/TPB/SHB/BVH cluster highest) — but correlating gate-mean against the
ablation-derived `delta_qlike` (from the 3rd attempt above) gave **Pearson r=0.13, Spearman
ρ=0.09** — near-zero and in the WRONG direction (higher gate slightly associated with news
*hurting* QLIKE more, not helping). Also: all gate values stay small (<0.09, std comparable to
mean) — the model mostly keeps news half-closed everywhere, so the between-ticker spread may
itself be more noise than signal.
**Why this matters:** even a fully learned, end-to-end differentiable gate (no external
pre-selection at all) does not agree with the internal ablation's notion of "which tickers
benefit from news." This is the 4th independent method (broad EDA, narrow EDA, internal
ablation, learned gate) to disagree with at least one other method — strengthens the
conclusion that stable, transferable per-ticker "news usefulness" may not exist in this
architecture/dataset, rather than any one method being flawed.
**How to apply:** if asked again to find "which VN30 tickers need news," don't just try a 5th
selection method — first ask whether the premise (a stable per-ticker property) is worth
continuing to test, e.g. via multi-seed stability of the gate itself, before building another
selection-based baseline.
