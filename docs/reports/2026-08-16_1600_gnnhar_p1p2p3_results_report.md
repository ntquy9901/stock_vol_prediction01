# GNNHAR (2308.01419) P1/P2/P3 — full converged results (3 seeds, DM)

Full run: 3 sweeps × 3 seeds {42,123,2026} × horizons {1,5,10,22}, 15-epoch cap (early-stop
patience 3, min 6), batched, on current VN30 data (33 tickers, refreshed to 2026-08-14). Sweeps:
MSE-2hop (reference), QLIKE-2hop (P1), QLIKE-1hop (P2). TS base `2026-08-16_141447_gnnhar`. QLIKE
uses one shared floor (1e-8) across every compared model. DM = HLN, HAC lag h−1, seed-ensembled.

## P1 — QLIKE training loss: FULL vs HAR (3-seed mean test QLIKE + DM)

| h | HAR | FULL(MSE) | FULL(QLIKE) | DM FULL(QLIKE) vs HAR |
|---|---|---|---|---|
| 1 | 0.4633 | 0.4589 | 0.4599 | −3.01, p=0.003* → FULL |
| 5 | 0.5503 | 0.5484 | 0.5486 | −1.21, p=0.225 → tie |
| 10 | 0.5933 | 0.5993 | 0.6008 | +2.45, p=0.014* → HAR |
| 22 | 0.6474 | 0.6735 | 0.6657 | +4.42, p<0.001* → HAR |

QLIKE-training vs MSE-training: comparable at h1/h5, narrows the gap at h22 (0.6735→0.6657); it is
not a long-horizon fix. FULL beats HAR significantly at h1; HAR wins h10/h22 — the paper's pattern
(deep-model edge at short horizon, HAR hard to beat long).

## Best trained variant vs HAR (QLIKE-2hop, 3-seed, DM) — the parsimonious models win

| h | HAR | FULL | best variant | best vs HAR (DM) |
|---|---|---|---|---|
| 1 | 0.4633 | 0.4599 | **lstm_only** 0.4553 | −5.52, p<0.001* → beats HAR |
| 5 | 0.5503 | 0.5486 | **minus_news** (LSTM+graph, no news) 0.5430 | −3.08, p=0.002* → beats HAR |
| 10 | 0.5933 | 0.6008 | lstm_only 0.5953 | +0.51, p=0.608 → tie |
| 22 | 0.6474 | 0.6657 | lstm_only 0.6614 | +4.48, p<0.001* → HAR |

A deep model beats HAR significantly at h1 and h5 on QLIKE — but the winning variant is the
parsimonious price-only LSTM (h1) or LSTM+graph-without-news (h5). The full model is dragged down by
the news/gate branches (below).

## Leave-one-out DM (QLIKE-2hop, 3-seed): which component contributes

FULL vs FULL−X; negative dm = FULL better = X contributes. * = p<0.05.

| h | −graph | −gate | −news(+gate) |
|---|---|---|---|
| 1 | −0.21 p=0.83 (ns) | +3.70 p<0.001* (news-gate does NOT help) | +9.41 p<0.001* (news does NOT help) |
| 5 | −0.59 p=0.56 (ns) | −0.28 p=0.78 (ns) | +6.58 p<0.001* (news does NOT help) |
| 10 | +2.01 p=0.04* (graph slightly negative) | +1.15 p=0.25 (ns) | +0.65 p=0.51 (ns) |
| 22 | +1.60 p=0.11 (ns) | +0.33 p=0.74 (ns) | +0.22 p=0.83 (ns) |

Under QLIKE-2hop on current data, the news branch and the per-ticker gate do not contribute at short
horizons (removing them improves QLIKE, significant at h1/h5); the graph is neutral-to-slightly
negative. This diverges from the earlier MSE-trained selective-news result (memory
`project_selective_news_gate_finding` / null-result pivot) and should be reconciled: differences are
the QLIKE loss, mini-batch optimization, and the 2026-08-14 data refresh. Needs a follow-up to
confirm whether news adds value under any current configuration.

## P2 — GAT depth: 1-hop vs 2-hop (QLIKE, 3-seed, DM) — contradicts the paper on VN data

| h | FULL 2hop | FULL 1hop | DM 2hop vs 1hop | 1hop vs HAR |
|---|---|---|---|---|
| 1 | 0.4599 | 0.4619 | −2.81, p=0.005* → 2hop | −1.65, p=0.098 (ns) |
| 5 | 0.5486 | 0.5507 | −0.27, p=0.787 → tie | −0.93, p=0.354 (ns) |
| 10 | 0.6008 | 0.6088 | −5.04, p<0.001* → 2hop | +3.67, p<0.001* → HAR |
| 22 | 0.6657 | 0.6663 | −0.02, p=0.981 → tie | +4.23, p<0.001* → HAR |

On VN30, 2-hop is significantly better than 1-hop at h1 and h10 and never worse — the opposite of
Zhang et al. (1-hop enough on DJIA; 2-hop over-smooths). Dropping to 1-hop also loses the significant
h1 win over HAR (p=0.003 → p=0.098). Conclusion: keep 2-hop on VN. (MAD-by-depth diagnostic
available via `model.gat_layer_outputs` + `mad.mad` for a future over-smoothing quantification.)

## P3 — regime split (calm 90% / turbulent 10%), QLIKE-2hop, 3-seed, DM FULL vs HAR

| h | calm HAR | calm FULL | DM calm | turb HAR | turb FULL | DM turbulent |
|---|---|---|---|---|---|---|
| 1 | 0.3828 | 0.3791 | −3.45, p<0.001* → FULL | 1.1875 | 1.1727 | −1.16, p=0.24 → tie |
| 5 | 0.4406 | 0.4238 | −13.86, p<0.001* → FULL | 1.5350 | 1.6553 | +5.23, p<0.001* → HAR |
| 10 | 0.4686 | 0.4603 | −6.61, p<0.001* → FULL | 1.7152 | 1.8441 | +6.12, p<0.001* → HAR |
| 22 | 0.5004 | 0.5036 | +1.29, p=0.20 → tie | 1.9696 | 2.1091 | +3.92, p<0.001* → HAR |

The most informative finding: on calm days (90% of obs) FULL beats HAR significantly at h1/h5/h10;
on turbulent days (top-10% by target volatility) HAR wins at h5/h10/h22. Turbulent QLIKE (1.2–2.1)
is ~3–4× calm (0.38–0.50), so the turbulent tail dominates the pooled average and turns the h5/h10
pooled result into tie/HAR. This is the exact effect the paper warns of (averages hide where a model
adds value) — but the sign on turbulent days is REVERSED vs the paper (there nonlinear spillover
helped in turbulence; on VN both models under-predict spikes and HAR under-predicts less).

## Overall conclusions

1. **A deep model beats HAR significantly at h1 and h5 on QLIKE (3-seed, DM)** — refines the earlier
   pooled parsimony-null. The winning variant is parsimonious (price-LSTM, or LSTM+graph without
   news); the full news+gate stack does not help and hurts at short horizons under this config.
2. **HAR remains best at h10/h22** (pooled) — parsimony holds long-horizon.
3. **2-hop ≥ 1-hop on VN** (contradicts the paper) — keep it.
4. **Regime split is the key analysis**: FULL wins the 90% calm regime (h1/h5/h10); the 10% turbulent
   tail (where HAR wins and both under-predict) dominates the pooled loss.
5. QLIKE-training helps mainly by narrowing the long-horizon gap, not reversing it.

## Follow-ups
- Reconcile the news no-lift here vs the earlier MSE-trained news-helps result (loss? batching? data
  refresh?).
- Report calm/turbulent separately in the paper (with the DM table above).
- Optional: MAD-by-depth number; rolling-origin (P4) robustness.
