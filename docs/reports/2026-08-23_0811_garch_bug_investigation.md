# GARCH(1,1) benchmark: bug or genuine? — adversarial investigation

## 1. VERDICT

**GENUINE (no units/scale/basis bug), with a PARTIAL caveat.** The large GARCH errors are *not* a
units, sqrt, annualization, floor, masking, or aggregation bug. The scale transform is correct and
the metrics reproduce the stored `result.json` exactly. The huge pooled MSE/R² is driven by a
**handful of tickers whose per-ticker GARCH fit lands on IGARCH (α+β = 1.000)**, causing the
frozen multi-step forecast to **diverge (grow ~linearly), not converge**, over the very long
(~1800-step) frozen test path. This is honest — if degenerate — GARCH behavior, amplified by the
"forecast the whole test window as one frozen path" design. The **paper's stated mechanism
("converges to the unconditional variance") is technically inaccurate** for the dominant blow-up
cases and should be corrected/disclosed.

Reproduced, not asserted: I rebuilt the real SP500 panel and recomputed GARCH — `mse=3.519216e-05`
vs stored `3.5190647e-05` (match to 5 sig figs).

---

## 2. What was checked (end-to-end, executed)

Files read:
- `scripts/garch_masked/compute_garch_masked.py` — `_garch_pred` (L78-102), `_harx_pred` (L105-112),
  `_metrics` (L54-59), `_dm` (L62-75).
- `submission/soict_lstm_gat/baselines.py` — `garch_forecast` (L85-135), scale transform (L114-129).
- `submission/soict_lstm_gat/metrics.py` — `mse/rmse/mae/qlike/r2`, `per_obs_qlike` shared floor.
- `baselines/2026-08-21_har_anchored_residual/code/masked_rich.py` — `build_masked_rich`, target
  panel (variance target), per-node scalers.
- `scripts/garch_masked/run_oos_suite.py` — HOSE/HNX/SP500 use the *identical* `garch_forecast` +
  `_garch_pred` + `_metrics` as VN30/VN100; only the file/price maps and the liquidity screen differ.

### 2.1 Scale / units are CORRECT (not the bug)
Target `parkinson_volatility` is a **variance** (~1e-4; AAPL mean 4.95e-4). `garch_forecast`
(`baselines.py:114-129`):
- `scale = sqrt(var)` ≈ 2.2e-2 → `pseudo_returns = sign·scale·100` ≈ O(1) (percent range),
- fit GARCH(1,1) on pseudo-returns → conditional-variance path in scaled space (~O(1)),
- `out = variance_path / 100**2 = /1e4` → back to raw variance (~1e-4).

Dimensionally exact: Var(pseudo_returns) = 100²·var, so dividing the variance forecast by 100²
recovers the raw variance. Confirmed empirically — for the **median** SP500 ticker the GARCH
forecast level is right: median true `2.91e-4`, median GARCH forecast `4.69e-4`, ratio **G/true =
1.53**. HAR-X median `3.13e-4`. No factor-of-10/100/sqrt error at the typical ticker.

### 2.2 Basis / floor / masking are apples-to-apples (not the bug)
GARCH and HAR-X are scored by the **same** `_pred_dict` over the **same** `tmask_te` observations,
the **same** per-node positivity floor `1e-2·t_mean` (`_garch_pred:90-101` vs `_harx_pred:111`), and
the **same** QLIKE floor `cfg.qlike_floor=1e-8`. `run_oos_suite._add_garch` asserts the recomputed
HAR-X QLIKE matches the stored value before writing GARCH (basis guard). No mismatch.

### 2.3 The actual mechanism (executed, per-ticker)
Rebuilt the real screened SP500 panel (498→440 nodes) and computed per-node GARCH forecasts:

| ticker | n_te | true mean | GARCH FC mean | HAR-X FC mean | train mean | G/true |
|--------|------|-----------|---------------|---------------|-----------|--------|
| MSCI | 1624 | 3.18e-4 | **8.31e-2** | 3.38e-4 | 6.71e-4 | **261×** |
| BLDR | 1624 | 7.60e-4 | 3.95e-2 | 7.40e-4 | 2.53e-3 | 52× |
| ISRG | 1624 | 3.20e-4 | 3.47e-2 | 3.39e-4 | 1.05e-3 | 108× |
| CPRT | 1624 | 2.62e-4 | 2.18e-2 | 2.85e-4 | 6.63e-4 | 83× |
| AOS  | 1624 | 2.40e-4 | 1.74e-2 | 2.68e-4 | 2.81e-4 | 73× |

- **Median ticker G/true = 1.53; mean = 5.60; only 23% of tickers have G/true > 3.**
- **Concentration: top-5 tickers carry 69.3% of total GARCH SSE; top-20 carry 89.8%.**
- The blow-up forecast (`8.3e-2`) is **~124× the ticker's own train mean** (`6.7e-4`) — so it is
  NOT the mean-fallback and NOT a level shift; it is a diverging conditional-variance path.

Direct parameter diagnosis (per-ticker GARCH fit, scaled space; `omega/alpha/beta`, unconditional
variance `ω/(1−α−β)` in raw units, and the multi-step forecast path):

| ticker | α+β | uncond var (raw) | train sample var | fc[1] | fc[300] | fc[1800] |
|--------|-----|------------------|------------------|-------|---------|----------|
| ISRG | **1.00000** | 7.8e+6 (∞) | 3.9e-6 | 1.98e-4 | 2.53e-3 | **1.42e-2** |
| CPRT | **1.00000** | 7.6e+6 (∞) | 1.7e-6 | 9.30e-4 | 3.21e-3 | 1.47e-2 |
| AOS  | **1.00000** | 7.0e+6 (∞) | 3.7e-7 | 9.05e-5 | 2.17e-3 | 1.26e-2 |
| AAPL | **1.00000** | 2.0e+0 (∞) | 1.1e-6 | 1.71e-4 | 6.22e-4 | 2.88e-3 |
| BLDR | 0.99505 | 2.9e-3 | 2.6e-5 | 1.61e-3 | 2.64e-3 | 2.94e-3 |
| VTR  | 0.99310 | 1.0e-3 | 3.5e-6 | 6.95e-4 | 9.63e-4 | 1.00e-3 |

**Mechanism:** long daily equity series routinely fit **IGARCH (α+β = 1 exactly)**. Under IGARCH the
k-step variance forecast has no finite unconditional limit — it grows ≈ linearly in k
(`fc ≈ σ²_last + (k−1)·ω`). `_garch_pred` forecasts `n_va + n_te` steps from train-end and keeps the
LAST `n_te` (`compute_garch_masked.py:99-101`); for SP500 that means the kept test forecasts sit at
steps ~200–1800, i.e. deep in the diverging tail (ISRG fc rises 2.0e-4 → 1.42e-2, ~72×). Those
inflated forecasts vs true ~3e-4 produce per-obs squared errors ~1e-3–1e-2 that, pooled over 1624
obs each, dominate the 714,520-obs mean. Even at ~1% of observations, 5 tickers set the MSE.

### 2.4 Why VN30 (1.4×) → SP500 (57×) grows
Not a scale artifact of "universe size" per se, but of **history length and count**:
- SP500 tickers have very long daily histories (AAPL 11,513 rows) → GARCH MLE lands on
  α+β=1 (near-unit-root) far more often than on short VN30 series.
- 440 nodes → many more chances to draw a degenerate IGARCH fit; MSE (a mean of squares) is
  dominated by the worst few, so more nodes ⇒ heavier right tail ⇒ larger pooled MSE and more
  negative R². HOSE sits between (fewer/shorter series than SP500, longer/more than VN30), and its
  h1/h5 MSE (156, 129 ×1e7) then drops at h10/h22 (29, 30) — consistent with horizon-dependent
  survivor/anchor counts, not a fixed level story.

---

## 3. Is it a code bug? (what a fix would/wouldn't change)

No scale/units/basis bug to fix. Two *methodological/robustness* choices produce the absurd
magnitude; both are defensible-to-question, neither is a miscomputation:

1. **Frozen ~1800-step forecast path** (`_garch_pred` forecasts `n_va+n_te` steps, keeps last
   `n_te`; `compute_garch_masked.py:99`). For h=1 the real target is 1-step-ahead, but it is scored
   against a k-step-ahead-from-train-end forecast with k up to ~1800. Under IGARCH this is the
   diverging regime. A GARCH benchmark that is "poor but not absurd" would instead use the
   **short-horizon** forecast (e.g. the h-step conditional forecast, or a flat unconditional level
   `ω/(1−α−β)` with a guard), rather than the far tail of a frozen path.

2. **No guard against α+β→1 / no variance targeting** (`garch_forecast:119-133`). The fallback
   only fires on non-finite/non-positive output (L131); a *finite but diverging* IGARCH forecast
   (1.4e-2) passes `isfinite` and `>0` and is kept. Options if a saner benchmark is wanted (do NOT
   apply without your decision): cap persistence (`α+β ≤ 0.999`), use variance targeting
   (`ω` fixed to sample-var·(1−α−β)), forecast only h steps (not n_va+n_te), or fall back to the
   train-variance mean when `α+β ≥ 1−ε`. Any of these would collapse the top-5 blow-ups and bring
   SP500/HOSE MSE down by ~1–2 orders of magnitude, while leaving VN30 (already benign) ~unchanged.

**Expected effect of a persistence cap / short-horizon fix (order-of-magnitude estimate):** removing
the ~5 IGARCH blow-ups (69% of SSE) would drop SP500 h1 GARCH MSE from ~3.5e-5 toward ~1e-5–1.5e-5
(R² from −38 toward roughly −5 to −10) — GARCH would still lose to HAR-X (as it should), but no
longer "absurdly."

---

## 4. Paper wording adequacy

The current text (`docs/paper/soict_harlstmgat_extended.tex:273`):
> "GARCH is a frozen-train forecast that converges to the unconditional variance, so its point
> error is large … where the volatility level shifts between train and test."

**Inadequate / technically inaccurate** for the cells that dominate the numbers:
- The blow-up tickers fit **IGARCH (α+β=1)**, where the forecast **does not converge** to a finite
  unconditional variance — it **diverges** over the long frozen horizon. Saying it "converges to the
  unconditional variance" mis-states the exact failure mode.
- The pooled error is **not a broad train/test level shift**; it is **~5 tickers (69% of SSE)** with
  degenerate near-unit-root fits amplified by a ~1800-step frozen forecast path. The "level shift"
  story is at best secondary.
- L395's SP500 note "consistent with a frozen-train forecast on illiquid series" is also off —
  SP500 is not illiquid; the driver is IGARCH divergence, not illiquidity.

**Recommendation (your call):** either (a) keep the current GARCH numbers but replace the mechanism
sentence with an accurate one — e.g. "GARCH(1,1) is fit per ticker and forecast as a frozen path
over the test window; for long equity series the fit is often near-integrated (α+β≈1), so the
multi-step forecast drifts upward and a few tickers dominate the pooled point error, explaining the
large MSE and negative R²" — and note the top-k SSE concentration; or (b) add a persistence cap /
short-horizon variant so GARCH is "poor but not absurd," and report that instead. Do not leave the
"converges to the unconditional variance" wording as-is.

---

## 5. Evidence / reproduction commands

- Panel + metric reproduction and per-ticker table: rebuilt `masked_rich.build_masked_rich` on the
  screened SP500 set (edge computation monkeypatched to identity — GARCH does not use edges), then
  `compute_garch_masked._garch_pred` / `_harx_pred` / `_metrics`. GARCH `mse=3.519216e-05` matches
  stored `results/masked_rich_floor1e2/sp500_h1/result.json` `metrics.GARCH.mse=3.5190647e-05`.
- Per-ticker parameter diagnosis via `arch.arch_model(..., rescale=False)` on each ticker's train
  Parkinson-variance series (same pseudo-return construction as `garch_forecast`).
- Env: base `python` with `arch 8.0.0`.

Note: the per-ticker parameter table (§2.3, second table) used a simple first-80% train split for
the parameter read-out, so its exact α/β differ slightly from the panel's anchor-subsampled train
series (e.g. MSCI reads α+β=0.968 there but blows up to 8.3e-2 in the panel fit) — the *mechanism*
(IGARCH → diverging long-horizon forecast) is identical and confirmed on ISRG/CPRT/AOS/AAPL, which
fit α+β=1.000 under both splits.
