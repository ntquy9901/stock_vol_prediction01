# Literature note — handling zero-valued / zero-range volatility observations (floor vs drop vs positive-proxy)

Date: 2026-08-31.

> **Verification status.** This note is compiled from domain/training knowledge. Live web verification was
> NOT possible this session: the WebSearch budget was exhausted (200/200 hard cap) and the alternative MCP
> search backend returned an out-of-balance error; academic PDFs did not parse via WebFetch (binary/compressed).
> **Every citation below must be re-opened and verified before use in the paper.** To run the full multi-agent
> deep-research (5-angle search → fetch → 3-vote adversarial verification → cited synthesis), raise
> `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` (or fund the MCP search) and re-request; the workflow is otherwise
> ready.

## 1. The problem
QLIKE loss `L(σ², ĥ) = σ²/ĥ − ln(σ²/ĥ) − 1` requires a strictly positive actual `σ²` and forecast `ĥ`
(the `ln` term). When the realized measure equals 0, QLIKE is `+∞` (undefined). Zero actuals arise mainly in
two settings that are relevant to this project:
- **Daily range estimators** (Parkinson `ln(H/L)²/(4ln2)`, Garman–Klass, Rogers–Satchell): on a day with
  `H = L` (limit-lock, halt, or no-trade), the intraday range is 0 → estimator = 0.
- **Illiquid / price-limit markets** (e.g. VN ±7/10%, Chinese A-shares ±10%): limit-lock days show `H = L`
  (zero intraday range) yet a large close-to-close move — genuinely high-volatility days, not calm ones.

## 2. Mainstream realized-volatility literature usually AVOIDS the zero
Canonical realized-variance forecasting work uses **intraday** realized variance = sum of squared intraday
returns, which is `> 0` by construction, so the zero-target problem does not arise and no explicit
floor/drop is discussed:
- Patton (2011), "Volatility forecast comparison using imperfect volatility proxies", *J. Econometrics*
  160(1) — the canonical QLIKE robustness reference; assumes a positive proxy.
- Andersen, Bollerslev, Diebold & Labys (realized-volatility foundations); Hansen–Lunde–Nason (Model
  Confidence Set) — QLIKE on positive intraday RV.

Implication: papers that "return 0" essentially do not exist in the QLIKE-evaluation literature, because a
literal 0 makes the loss undefined. The zero is avoided upstream (positive proxy), not fed into the loss.

## 3. Three treatments when zeros DO occur (no single universal convention)
| # | treatment | who / where | note |
|---|---|---|---|
| (a) | **Floor / ε** — clamp actual (and forecast) to a small positive constant before the loss | common in software implementations of QLIKE / log-loss | numerical guard against `log(0)`; what this project does (`QLIKE_FLOOR = 1e-8`, applied to BOTH y and ŷ on the same basis) |
| (b) | **Drop / screen** — remove the degenerate (zero-range / limit / no-trade) observations from the evaluation (and sometimes estimation) | range-estimator & price-limit-market literature | methodological choice = "forecast only real trading days"; biases the sample away from limit-lock high-vol days (must be disclosed) |
| (c) | **Positive proxy** — use intraday RV or an overnight-augmented estimator that stays `> 0` on limit days | intraday-RV literature; Yang–Zhang / GK-YZ for daily | avoids the zero at the source |

## 4. Range-estimator & price-limit specifics
- Range / CARR literature (Alizadeh, Brandt & Diebold 2002, *J. Finance*; Chou 2005 CARR, *JMCB*; Molnár
  2012, *Applied Financial Economics*) works with `log(range)`, which breaks at zero range. On the liquid
  indices/large-caps these studies use, zero range is rare; on illiquid data the standard response is to
  screen or add ε.
- Price-limit markets (Chinese A-share ±10% studies): limit-lock days have `H = L` but a large gap move, so
  **range estimators under-estimate** volatility on exactly those days. Typical handling = exclude limit-hit
  days, treat them specially, or switch to a close-to-close / overnight-augmented measure. This matches the
  project's own EDA (an overnight-bearing estimator, `yang_zhang_n20`, is non-zero on 82–97% of the
  zero-range days that Parkinson floors).

## 5. Mapping to this project (VN, Parkinson-variance target)
- **What the code does now:** the zero is a valid stored feature value; at the LOSS it is handled by the
  shared positivity floor (treatment (a)). `zero_range_flag` / `zero_volume_flag` are carried but not yet used
  to screen.
- **Both (a) and (b) are standard, not mutually exclusive.** They differ only in the assumption "do we forecast
  limit-lock days or not." Recommendation: keep the floor as the numerical guard, and evaluate a screened
  variant as an ablation (drop only the degenerate-target `zero_range` rows — NOT the cosmetic-for-Parkinson
  `open_close_outside` / `split_jump` rows, since `ln(H/L)` is invariant to those), recompute causal features
  AFTER the drop (to keep rolling-window semantics), and disclose the sample-selection.
- **Scale reminder (grounded in this repo's data):** `dirty_flag` is ≈99% `zero_range`; dropping "all dirty"
  removes ~3.3% of VN100 but ~45% of HNX — so a drop policy is minor on VN100/VN30/SP500 and a major sample
  intervention on HNX/HOSE.

## 6. Open items to verify when search is available
- Specific VN / emerging-market volatility papers and their explicit limit-day treatment (drop vs special
  handling) — not verifiable this session.
- Whether any QLIKE study documents a specific floor value convention (vs ad-hoc ε).
- Precise wording of the Chinese A-share limit-day exclusion rules.

## References (from knowledge — RE-VERIFY before citing)
Patton (2011), *J. Econometrics* 160(1); Andersen–Bollerslev–Diebold–Labys (realized volatility);
Hansen–Lunde–Nason (MCS); Alizadeh, Brandt & Diebold (2002), *J. Finance*; Chou (2005) CARR, *JMCB*;
Molnár (2012), *Applied Financial Economics*; Chinese A-share price-limit volatility literature.
