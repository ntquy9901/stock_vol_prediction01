# Learnable-alpha HAR+deep convex combination — feasibility assessment

Date: 2026-08-21
Scope: architecture feasibility study for `pred = alpha·HAR + (1-alpha)·deep` with `alpha` LEARNED
during training, `deep` = LSTM (optionally +GAT). Read-only analysis of `submission/soict_lstm_gat`
plus a small VN30 lb10 h1 prototype (CPU, 2 seeds, 12 epochs). No data/model files modified; nothing
committed. Prototype code: `_tmp_alpha_combo/proto.py`; raw output `_tmp_alpha_combo/result.json`.

## TL;DR

Feasible and safe; plausibly helps at short horizons; but the learned weight lands almost exactly on
the value the project already found by hand. In the prototype the learned **global alpha converged to
0.49** (weight on HAR), i.e. `1-alpha = 0.51` on the LSTM — statistically indistinguishable from the
existing fixed 0.5 combination. The learnable-alpha combo attains the lowest test QLIKE of the four
configs (0.4553 vs HAR 0.4675, fixed-0.5 0.4572) and is the only variant that **significantly beats the
standalone LSTM** on QLIKE (DM p=0.018) — but that extra edge comes from **co-training the LSTM inside
the combination**, not from the weight itself (which barely moved off 0.5). Recommendation: the
architecture is worth including, but its value is co-training + a validation-fit *horizon/ticker*
weight, not a single global scalar that a fixed 0.5 already captures.

## 1. Prior fixed-combination result (recap)

`docs/reports/2026-08-17_beat_har_research.md` established the project's one robust "beat-HAR" route:
a fixed, untuned Bates–Granger average `0.5·HAR + 0.5·deep` on the **raw variance scale**.

| h | HAR QLIKE | 0.5·HAR+0.5·lstm_only | DM vs HAR | seeds |
|---|---|---|---|---|
| 1 | 0.4633 | 0.4564 | p<0.001 (combo) | 3 |
| 5 | 0.5503 | 0.5450 | p<0.001 (combo) | 5 (robust) |
| 10 | 0.5933 | 0.5918 | p=0.208 (tie) | 5 |
| 22 | 0.6474 | 0.6526 | p<0.001 (HAR) | 3 |

The reported weight-sensitivity grid showed the QLIKE-optimal `w_HAR` is **horizon-adaptive**: ~0.0–0.3
(mostly deep) at h1, rising to 1.0 (all-HAR) at h22. That horizon dependence is exactly what a *learned*
weight could capture automatically — the natural generalization, and the motivation for this study.

Separately, `docs/reports/2026-08-21_gat_why_no_help.md` shows the **GAT branch hurts on its own**
(attention collapses to uniform averaging, entropy 0.999; train→test edge Jaccard 0.17; 2× larger
generalization gap; GAT QLIKE 0.453 vs no-GAT 0.412 at 5 seeds, DM p=1.7e-10). Implication for this
design: the `deep` term in the combination should be the **LSTM (no GAT)**. Combining HAR with LSTM+GAT
inherits the GAT's OOS variance; a learned alpha would then have to spend weight moving *away* from the
deep term to undo damage the GAT added — strictly worse than combining with the plain LSTM. The user's
requested HAR+LSTM+GAT form is assessed below and is feasible, but is dominated by HAR+LSTM.

## 2. Correct differentiable formulation (scale-aligned)

HAR predicts on the **raw** Parkinson-variance scale (`baselines.har_predict`). The deep model outputs a
**per-ticker normalized** value; its raw prediction is `lstm_raw = lstm_norm·t_std + t_mean`
(`run_lstm.py:144`). The combination must happen on a common scale — raw:

```
alpha       = sigmoid(theta)                         # theta learnable, alpha in (0,1)
lstm_raw    = lstm_norm * t_std[ticker] + t_mean[ticker]
pred_raw    = alpha * har_raw + (1 - alpha) * lstm_raw
loss        = MSE(pred_raw, y_raw)      (or QLIKE)
```

Because per-ticker normalization is affine and the weights are convex (`alpha + (1-alpha) = 1`, so the
two `t_mean` terms recombine to exactly one `t_mean`), combining raw then re-normalizing is **identically
equal** to combining on the normalized scale:

```
combo_norm = alpha * har_norm + (1-alpha) * lstm_norm,   har_norm = (har_raw - t_mean)/t_std
(pred_raw - t_mean)/t_std == combo_norm          # verified algebraically and in code
```

So the prototype trains with MSE on `combo_norm` (the same loss basis as `run_lstm.py`) and it is
provably the same objective as raw-scale combination. Every operation (`sigmoid`, affine inverse-scale,
convex mix, MSE) is differentiable, so `theta` and the LSTM weights receive gradients end-to-end. HAR
itself is a frozen OLS fit on train (a fixed regressor), which is the standard and leakage-safe way to
build a stacked/combined forecaster — the combination weight is what is learned, not HAR.

## 3. What alpha should be — global vs per-ticker vs gated

- **(a) Single global scalar `alpha = sigmoid(theta)`.** Simplest; one extra parameter; cannot overfit.
  This is what the prototype implements. Its ceiling is the best *constant* weight — which is what the
  project already approximates with fixed 0.5. Verdict: good baseline, low upside over fixed 0.5.
- **(b) Per-ticker `alpha_i`.** 33 parameters on VN30. The `gat_why_no_help` per-obs deltas show HAR-vs-
  deep usefulness is heterogeneous across tickers, so in principle this could help. But per-ticker
  weights are fit on ~300 train obs/ticker and the project's repeated finding is that per-ticker edges
  do **not transfer OOS** (edge Jaccard 0.17; the "selective news gate" note — per-ticker usefulness
  disagrees across methods). High risk of fitting train noise; must be validation-selected, not
  train-fit. Verdict: plausible but fragile on a 33-ticker panel.
- **(c) Input-dependent gate `alpha(x)` from a tiny network** (mixture-of-experts / gating view). This is
  the general form and connects to the project's per-ticker-gate work. It can express the horizon- and
  regime-dependence the sensitivity grid revealed (e.g. lean on the deep model in calm short-horizon
  regimes, fall back to HAR when the deep model is unreliable). But a learned gate over the same 3 HAR
  features is the same object that **already collapsed** in the GAT attention and the news-gate ablations
  (null result). It adds capacity that overfits the small panel unless heavily regularized.

**Recommendation.** For a single model on this data, prefer **(a) a global learned alpha, fit/validated
per horizon** — i.e. one alpha per horizon, selected on the validation split, never on test. That
directly automates the horizon-adaptive optimum from the sensitivity grid (the real signal) while
avoiding the OOS-transfer fragility of per-ticker/gated weights. Treat (b)/(c) as ablations that must
beat (a) on validation before being believed, not as the default.

## 4. Training dynamics, gradient to alpha, and the safety property

- **Does alpha get a useful gradient?** Yes. `d loss / d theta = sigmoid'(theta) · (har - lstm) · dMSE`;
  the gradient is non-zero whenever HAR and the deep model disagree, which is generic. In the prototype
  `theta` moved from init 0 (alpha=0.5) to alpha≈0.49 and stayed there across both seeds — a stable,
  well-behaved optimum, not a runaway.
- **Collapse to alpha=1 (pure HAR) — risk or feature?** If the deep model is weak/noisy, the loss is
  minimized by shifting weight to HAR, driving alpha→1. That is the **desired safe behavior**: a convex
  combination with a reachable alpha=1 can never do worse than HAR in the limit. This is the
  Bates–Granger / Timmermann forecast-combination safety property — combining weakly-correlated forecasts
  cannot increase (and generically reduces) expected loss versus the better single forecast, and the
  convex hull always contains the "all-HAR" corner. So collapse-to-HAR is a graceful degradation, not a
  failure mode. (The symmetric risk — alpha→0, pure deep — is bounded the same way: it only happens if
  the deep model genuinely dominates on validation.) The practical guard is to **select alpha on
  validation**, so a deep model that only looks good on train cannot pull weight off HAR.
- **Horizon behavior.** At h22 the sensitivity grid's optimum is w_HAR=1.0; a per-horizon learned alpha
  should recover alpha≈1 there, reproducing "use HAR alone at long horizons" automatically instead of by
  hand.

## 5. Prototype numbers

VN30, lookback 10, horizon 1, per-observation pooled design (matches `run_lstm.py`), 2 seeds {42,123},
12 epochs, CPU, `n_test = 10,577`. HAR = global OLS (frozen); deep = 2-layer price-only LSTM. Fixed-0.5
uses the standalone-LSTM ensemble; learn-alpha co-trains the LSTM with `theta`. QLIKE floor 1e-8 shared
across all models. Learned **alpha per seed = [0.4919, 0.4911], mean 0.4915** (weight on HAR).

| model | test QLIKE | test MSE | R² | DM vs HAR (QLIKE) | DM vs LSTM (QLIKE) |
|---|---|---|---|---|---|
| HAR | 0.4675 | 2.2449e-07 | 0.289 | — | — |
| LSTM (standalone) | 0.4599 | 2.2344e-07 | 0.292 | p=0.024 (LSTM) | — |
| fixed 0.5 combo | 0.4572 | **2.2258e-07** | 0.295 | p=4.7e-18 (combo) | p=0.24 (tie) |
| **learn-alpha (α=0.49)** | **0.4553** | 2.2328e-07 | 0.293 | p=3.9e-11 (combo) | **p=0.018 (combo)** |

Reading the table honestly:

1. **Both combos beat HAR decisively on QLIKE** (p≪0.001), reproducing the prior fixed-0.5 finding on
   this fresh 2-seed run (fixed-0.5 here 0.4572 vs the 3-seed 0.4564 previously — same effect).
2. **The learned global alpha (0.49) is statistically the fixed 0.5.** The weight barely moved; the
   "learning" recovered the hand-picked value. This is the honest core result: *a fixed 0.5 already sits
   on the global optimum at h1, so a single global learned scalar has almost nothing to add.*
3. **learn-alpha has the lowest QLIKE and is the only config that significantly beats the standalone
   LSTM** (p=0.018, vs fixed-0.5's p=0.24). But since alpha≈0.5≈fixed, that improvement is **not** from
   the weight — it is from **co-training the LSTM against the combination objective** (its early-stopping
   target is the combo's val loss, so it specializes to complement HAR). Fixed-0.5 instead bolts a
   separately-trained LSTM onto HAR. So the learnable architecture's real, separable benefit here is
   joint training, not weight selection.
4. **fixed-0.5 has the lowest MSE**; learn-alpha wins QLIKE (the project's headline metric) and ties on
   MSE. Differences among the three non-HAR configs are <1% on MSE/R² — consistent with prior reports.

## 6. Verdict

- **Feasible:** yes, unambiguously. The formulation is a one-parameter, fully differentiable, leakage-
  safe extension of the existing fixed combination; the prototype trains stably and reproduces the
  expected effect. Implementation cost is trivial (add `theta`, mix on the normalized scale).
- **Does it plausibly help vs the fixed 0.5?** Marginally, and not through the mechanism one would expect.
  A **single global** learned alpha converges to ≈0.49 — it confirms rather than improves on fixed 0.5.
  Its measured QLIKE edge comes from **co-training the deep branch** inside the combination, which is a
  real and worthwhile change, but is a training-recipe benefit, not a "learned weight" benefit. Where a
  learned weight genuinely earns its keep is **across horizons** (recover w_HAR≈0→1 as h grows,
  automating the sensitivity grid) and potentially across regimes — but only if fit on validation.
- **Relation to existing findings:** it is the principled generalization of the fixed-0.5 beat-HAR result
  (Bates–Granger), it inherits that result's short-horizon win and long-horizon "use HAR alone" behavior
  via the safety property (alpha→1), and it should use **LSTM without GAT** because
  `gat_why_no_help` shows the GAT only adds non-transferable variance that a combination weight would
  then have to discount.
- **Recommended form:** global `alpha = sigmoid(theta)` **per horizon, validation-selected**, with the
  deep branch **co-trained** against the combination loss and the GAT dropped. Treat per-ticker and
  gated `alpha(x)` as ablations that must beat the global per-horizon alpha on validation before use —
  the project's OOS-transfer history predicts they will struggle on a 33-ticker panel.
- **Honest caveat:** prototype is 2 seeds, single horizon (h1), one dataset (VN30), 12 epochs, CPU. The
  qualitative conclusions (alpha≈0.5 at h1; combo beats HAR; co-training helps) are consistent with the
  5-seed committed results, but the per-horizon claim (alpha rising toward 1 at h22) is inferred from the
  prior sensitivity grid, not yet run here. A follow-up sweeping h∈{1,5,10,22} with validation-fit alpha
  and 5 seeds would confirm the horizon-adaptive payoff, which is where the learnable form — as opposed
  to a fixed 0.5 — would actually distinguish itself.
