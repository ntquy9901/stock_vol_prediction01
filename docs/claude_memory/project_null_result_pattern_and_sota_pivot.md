---
name: project-null-result-pattern-and-sota-pivot
description: "~10 null results across news/graph/loss variants, THEN per-ticker isolated gate broke the streak (new best QLIKE/R2) but still doesn't match independent per-ticker usefulness signal"
metadata:
  node_type: memory
  type: project
  originSessionId: 4f7cf132-7896-4bf1-8313-3063fa32630a
  modified: 2026-07-26T16:01:49.661Z
---

As of 2026-07-26, roughly 10 independent architectural/feature variants have all failed to beat
the HAR-only baseline on directional accuracy: dual-group news, macro news, gated cross-attn,
selective gate, top3 gate, ablation-derived gate, REST-TS, alignment loss, pure-market,
market-fallback, latent-noise, a 12-new-source dual-group panel rebuild, and (most recently) a
directed volatility-spillover graph + QLIKE-augmented loss baseline
(`2026-07-26_spillover_qlike_baseline`).

**SOTA research finding (2026-07-26):** every prior baseline kept 2 things fixed — the inter-stock
graph (`src/lstm_gat_hybrid/graph_correlation.py`, always symmetric/same-day Pearson correlation)
and the training loss (always plain MSE, QLIKE only used for eval). 2025-2026 literature (Zhang,
Pu, Cucuringu & Dong, IJF 2025; Chi et al., J. Forecasting 2026) identifies exactly these 2 as the
SOTA gap: directed volatility-spillover graphs + QLIKE training loss. Implemented both together in
`2026-07-26_spillover_qlike_baseline` — still null (Test DirAcc 68.23% vs dual-group's 68.25% on
the same panel; R²/QLIKE/RMSE all within noise).

**Why:** this rules out "the graph construction or loss function was the missing piece" as an
explanation for the repeated null results, on top of ruling out "the news feature wasn't rich
enough" (already tried: raw PhoBERT embeddings, PCA-reduced, dual-group, macro broadcast, gated
attention, selective/top3 filtering).

**How to apply:** before proposing an 11th architectural variant, question the upstream premise
first — is 5-day-ahead daily Parkinson volatility on VN30 predictable beyond HAR's own
autocorrelation at all, given only daily-bar OHLCV + Vietnamese news? Concrete next steps
suggested (not yet tried): (a) an honest ceiling/headroom check (e.g. an oracle model with future
HAR features, to see if ANY model could beat HAR-only given current features), (b) intraday/
higher-frequency data if available — daily-bar granularity itself may be the limiting factor, not
the model architecture. See [[project_selective_news_gate_finding]] for the earlier (2026-07-25)
version of this same pattern before the SOTA pivot was tried.

**UPDATE 2026-07-26 (later same day) — first genuinely positive result, with an important caveat.**
`2026-07-26_per_ticker_news_gate_baseline`: gave EACH ticker its own free scalar gate parameter
(not shared across tickers, unlike `gated_crossattn`'s `gate_mlp`) so `∂loss/∂gate_logits[i]` is
PROVABLY isolated to ticker i's own prediction error (verified by a direct gradient-perturbation
test, not just architectural reasoning). Result: Test QLIKE=0.5497 (new project-best, beats
gated_crossattn's 0.557), R²=0.7159 (ties/edges its 0.7157 record), DirAcc=68.76% and
RMSE=0.002635 both beat the same-panel dual-group baseline (68.25%/0.002651) on all 4 metrics —
the first clear win after ~10 null results.

**But:** correlating the learned per-ticker gate values against the independent ablation's
`delta_qlike` (`results/ablation_derived_ticker_classification.json`) gave Pearson r=0.14,
Spearman ρ=0.07 — statistically indistinguishable from the OLD shared-weight gate's r=0.13, wrong
direction. Mean gate for the ablation's NEWS_ON tickers (0.561) vs NEWS_OFF (0.572) showed no
real difference. So even with gradient isolation proven, the mechanism still does NOT discover
the same "which ticker benefits from news" pattern the independent ablation found — this rules
out "shared weights caused the disagreement" as an explanation. The performance gain likely comes
from some other benefit of a per-ticker-adjustable scaling knob (regularization-like effect), not
from correctly identifying true per-ticker news causality. Also: gates were still moving >0.1 for
several tickers between epoch 9 and 10 — not converged within the 10-epoch cap, so current gate
values are a mid-training snapshot, not a stable readout.

**How to apply (updated):** if asked "which VN30 tickers actually benefit from news," the honest
answer as of 2026-07-26 is still "no method has found a stable, cross-validated answer" — 5
independent methods now disagree (broad EDA, narrow EDA, internal ablation, shared gate, isolated
gate). But the per-ticker-gate ARCHITECTURE itself is worth keeping/reusing regardless — it's the
best QLIKE/R² result in the project's news-fusion lineage, independent of whether its per-ticker
values mean what was hoped.

**UPDATE 2026-07-26 (same evening) — trained 10 more epochs (11-20), gate values are NOT stable.**
Resumed from the epoch-10 checkpoint. Aggregate metrics plateaued (QLIKE ticked down further to
0.5473, new best; DirAcc/R²/RMSE flat within noise) — but individual ticker gate values swung by
0.3-0.4 between epoch 10 and 20 (BID 0.26→0.65, HDB 0.77→0.43, VNM 0.35→0.63 — enough to flip
which "half" a ticker would be classified into). Epoch10-vs-epoch20 gate correlation across all
32 tickers: r=0.79 (general ranking roughly holds) but specific outliers are unreliable.
Correlation with the independent ablation's delta_qlike improved slightly (r=0.14→0.28) but still
not significant (p=0.12). **Conclusion: a single-epoch gate snapshot must not be read as a
settled per-ticker signal** — the aggregate performance gain is real and reproducible-so-far, but
which specific tickers look "high gate" depends on exactly which epoch you stop at. A stability
check (multiple random seeds) would be needed before trusting individual ticker values, not yet
done.

**UPDATE 2026-07-26 (same evening) — trained to epoch 40 total. Gate DOES converge with enough
epochs, but overall performance peaked earlier (~epoch 20) and degraded slightly afterward.**
Epoch-to-epoch40 gate correlation: r=0.69 (epoch10), r=0.95 (epoch20), r=0.98 (epoch30) — clean
monotonic convergence, i.e. more training genuinely stabilizes the per-ticker gate values.
Correlation with the independent ablation's delta_qlike also strengthened: r=0.14 (epoch10) →
0.28 (epoch20) → 0.35, p=0.053 (epoch40) — borderline significant, the best agreement seen across
all 5 methods so far, though still not conclusively significant. BUT aggregate test QLIKE got
WORSE after epoch 20 (0.5473 best at ep20 → 0.5612 at ep30 → 0.5613 at ep40) and val QLIKE moved
the same direction (0.691→0.700) — a real mild degradation on both splits, not just test noise.
**Practical trade-off:** the epoch-20 checkpoint has the best raw performance; the epoch-40
checkpoint has the most stable/interpretable gate ranking. Final epoch-40 gate ranking (high =
model relies on news most): GAS, NVL, SAB, MBB, SHB, VCB, ACB, BCM, SSI, VJC top; PLX, STB, VPB,
BVH, HPG, PDR, VIB, TCB, HDB, GVR bottom. Neither checkpoint should be discarded — keep both,
pick based on whether the goal is performance or interpretability.
