# Why the QLIKE-loss edge vanishes at h5/h10 (per-cell log analysis) + improvement directions

Per-cell (ticker×date) decomposition of the **QLIKE-loss** test forecasts (`qlike_anchor`, anchor=none)
on SP500 and VN100, comparing h1 (edge present) vs h5/h10 (edge gone). LSTM/VolGA are QLIKE-trained;
HAR-X is the same OLS. Per-cell QLIKE split by realized-volatility decile (D0=calmest … D9=most
volatile). Source cells: `results/qlike_anchor/cells/cells_{sp500_clean,vn100}_qlike_none_h*.parquet`.

**Decile (D0–D9) definition:** every test cell is one stock on one day. We sort all cells by that
day's *realized* Parkinson variance and cut into ten equal-size groups. **D0** = the 10% of
stock-days with the lowest realized volatility (calmest); **D9** = the 10% highest (spike days). This
separates "normal days" from "storm days" so we can see which model forecasts better in each regime.

## Finding 1 — Why LSTM does not beat HAR-X at h5/h10: the edge is calm-cell only; high-vol cells flip
The deep model's QLIKE advantage over HAR-X lives entirely in the **low-volatility deciles**, and it
holds at every horizon. What changes with horizon is the **high-vol deciles**: at h1 the deep model
wins (or ties) them; at h5/h10 it *loses* them, and those losses cancel the calm-cell wins.

SP500, per-decile QLIKE diff **LSTM − HAR-X** (negative = LSTM better):
| decile | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 |
|---|---|---|---|---|---|---|---|---|---|---|
| h1 | −.072 | −.064 | −.059 | −.047 | −.048 | −.039 | −.057 | −.043 | −.025 | −.007 |
| h5 | −.180 | −.113 | −.073 | −.040 | −.013 | **+.012** | **+.036** | **+.060** | **+.080** | **+.129** |
| h10 | −.218 | −.135 | −.087 | −.046 | −.011 | **+.021** | **+.053** | **+.082** | **+.107** | **+.152** |

VN100 shows the same flip (h1 mostly negative across deciles → h5/h10 negative on D0–D2, **positive
on D4–D9**). At h1 the deep edge is broad; at h5/h10 it is confined to D0–D4 and reversed on D5–D9.

- **Date win-rate stays high** (SP500 LSTM<HAR-X on 65% of dates at h1, still 68–69% at h5/h10), but
  the *mean* QLIKE edge collapses — a handful of high-volatility dates/cells (D8–D9) where HAR wins
  big drag the average and are exactly what the date-clustered DM keys on ⇒ DM n.s. at h5/h10.
- **Mechanism:** the h-step-ahead variance is a mean-reverted quantity at longer h. HAR's
  long-memory weighted average nails the mean-reversion of recently-elevated (high-vol) nodes; the
  deep model carries short-term persistence/momentum and **over-forecasts** those nodes at h5/h10.
  Its durable skill is on the **calm baseline** (predicting the low-variance floor), which QLIKE
  weights heavily (D0 QLIKE ≈ 0.7–1.0) and which persists across horizons.
- **A global HAR-X anchor does NOT fix it** (verified): anchor=harx QLIKE is worse than anchor=none
  at every horizon/market, because anchoring drags the deep model toward HAR-X on the calm cells
  where it wins. The fix must be **regime-selective**, not global.

## Finding 2 — Why VolGA does not beat LSTM (graph unhelpful) at h5/h10: short-lived spillover, extreme-cell noise
The directed volume→Parkinson graph is a **contemporaneous / short-horizon** spillover signal.

SP500, per-decile QLIKE diff **VolGA − LSTM** (negative = graph better):
| decile | D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 |
|---|---|---|---|---|---|---|---|---|---|---|
| h1 | −.009 | −.005 | −.004 | −.004 | −.003 | −.003 | −.002 | −.001 | +.002 | **+.025** |
| h5 | +.001 | +.002 | +.002 | +.001 | .000 | −.001 | −.001 | −.001 | −.001 | **+.017** |
| h10 | −.016 | −.014 | −.013 | −.011 | −.009 | −.008 | −.005 | −.001 | +.007 | **+.051** |

- At h1 the graph gives a small broad gain but is already **noisy on the extreme decile** (D9 +.025).
- At h5 the broad gain evaporates (near zero across deciles) and the top decile dominates the diff
  (84% of it) — the graph adds noise on the most volatile nodes and nets slightly *worse*.
- At h10 the graph helps the calm/mid deciles again but the extreme-cell noise (D9 +.051) is large;
  the net is tiny and dominated by D9. VN100 is the same shape (net ≈ 0, sign flips by horizon).
- **Mechanism:** same-day cross-sectional co-movement (volume-shock spillover) predicts *next-day*
  turbulence but washes out against a 5–10-day forward average; on the most volatile nodes the
  spillover is itself most volatile, so the graph injects variance exactly where it cannot help.

## Finding 3 — Why SP500 still beats HAR-X at h22 but VN does not (same mechanism, different magnitude)
Both markets follow the same shape at h22 (deep wins the calm deciles, loses the volatile ones). The
difference is the *size* of the calm-cell win, which on SP500 is large enough to outweigh the
volatile-cell loss.

LSTM − HAR-X QLIKE at **h22**, calm half (D0–D4) vs volatile half (D5–D9):
| panel | mean LSTM−HARX | calm half (D0–D4) | volatile half (D5–D9) | D0 diff | date win-rate |
|---|---|---|---|---|---|
| SP500 | **−0.0261** (sig, p=0.010) | **−0.148** | +0.096 | −0.294 (1.151→0.858) | 76% |
| VN100 | +0.0006 (n.s.) | −0.059 | +0.059 | −0.172 (1.906→1.734) | 56% |

- On SP500 the deep model's calm-cell advantage (−0.148) is ~2.5× VN100's (−0.059) and **exceeds** the
  volatile-cell loss (+0.096) ⇒ net negative and significant. On VN100 the two halves are equal
  (−0.059 vs +0.059) ⇒ they cancel to ≈0.
- **Why the calm edge is bigger on SP500:** on the calmest decile HAR-X massively over-forecasts
  (median forecast ≈ 3.7× realized), and the QLIKE-loss deep model corrects this floor much better —
  cutting D0 QLIKE by 25% (1.15→0.86) versus only 9% on VN100 (1.91→1.73). SP500's large cross-section
  of very liquid, persistently-low-vol large-caps makes the calm floor highly predictable; VN's
  thinner, noisier calm days give the deep model less durable calm-cell signal.
- **Plus statistical power:** SP500 has 320k cells / 667 dates and a 76% date win-rate, so a −0.026
  mean edge clears DM significance; VN100 (46k cells, 56% win-rate, ≈0 mean) cannot.

So SP500's h22 win is not a different phenomenon — it is the same calm-cell edge, just large enough
(and on a big enough sample) to survive the volatile-cell losses that erase it on the smaller,
noisier VN panels.

## Improvement directions (grounded in the two failure modes)
1. **Volatility-regime-conditional blend (largest lever).** Keep the deep model on low-vol nodes
   (D0–D4, where it wins at all horizons) and shrink toward HAR-X only on high-vol nodes (D5–D9) at
   long horizons, via a gate on the node's current realized-vol level (learned convex weight, or a
   hard regime split). This preserves the calm-cell edge that a *global* HAR-X anchor destroys.
2. **Regime-selective mean-reversion prior.** Add a shrink-to-HAR / target-decay term that activates
   only for currently-elevated nodes, correcting the deep model's over-forecast of mean-reverting
   spikes at h5/h10 without touching the calm cells.
3. **Horizon-gated graph.** The graph earns its keep at h1 and adds extreme-cell noise at h5/h10 —
   scale the GAT branch's contribution down with horizon (weight→0 as h grows), or down-weight edges
   *into* high-vol nodes. Alternatively swap the contemporaneous edge for a longer-window /
   lower-frequency spillover when forecasting long horizons.
4. **Tail-robust evaluation/training on high-vol cells.** The broad calm-cell edge (65–69% of dates)
   is erased in the mean by a few D9 dates; a winsorized/robust QLIKE (or down-weighting D9 in the
   loss) would both reflect the typical-day skill and reduce the deep model's incentive to chase
   spikes. This is an eval/loss choice; (1)–(3) are the modelling levers for a genuine forecast gain.

**Net:** the QLIKE-loss deep/graph edge is a *calm-node, short-horizon* effect. It fades at h5/h10
because (a) the deep model over-forecasts mean-reverting high-vol nodes and (b) the contemporaneous
graph signal does not persist to multi-day-ahead averages. A vol-regime-gated deep+HAR blend with a
horizon-gated graph targets both directly.
