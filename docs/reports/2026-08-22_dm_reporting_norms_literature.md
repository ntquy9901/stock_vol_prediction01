# DM / forecast-comparison reporting norms in the volatility literature

Date: 2026-08-22
Scope: sanity-check the paper's Diebold–Mariano (DM) reporting against field practice, and
recommend a simpler presentation for a SOICT-style conference paper.

## What the paper currently does

Date-clustered DM (one loss differential per calendar date, HLN small-sample correction) run
across THREE loss families (QLIKE, squared error, absolute error), plus a Model Confidence Set
(MCS) and block-bootstrap confidence intervals, per horizon, per panel.

## Method note

The two live web-search back-ends were out of budget/balance this session, so evidence was
gathered by fetching primary sources directly (arXiv HTML mirror, author-hosted PDFs, arXiv API)
and extracting text locally. Five sources were read at first hand; two (Corsi 2009 original HAR;
Branco–Rubesam–Zevallos "does anything beat linear models?") could not be retrieved in full and
are noted as such rather than quoted.

---

## 1. Is DM standard practice? Yes — but MCS/relative-loss framing is at least as common

| Work | DM run? | MCS run? | How significance is shown | Loss families |
|---|---|---|---|---|
| Patton 2011 (JoE), methodology | Notes DM/West are applicable under a robust proxy | — | (theory paper) | Defines the *robust* class; MSE & QLIKE are the two standard members |
| Bollerslev–Patton–Quaedvlieg 2016 (HARQ, JoE) | Reality-Check style bootstrap test vs benchmarks | Not prominent | **Relative loss ratios vs HAR, best in bold**; index tested via stationary-bootstrap Reality Check; cross-section = average/median ratios | **MSE + QLIKE** |
| Christensen–Siggaard–Veliyev 2026 (arXiv 2601.13014), ML vs HAR | Yes | Yes | DM **rejection-frequency notation embedded in the MSE table caption**; MCS shown in a **separate figure** | **MSE only** |
| GNNHAR, Zhang et al. 2023 (arXiv 2308.01419) | Yes | Yes | **Asterisk = best, dagger = in 5% MCS on the main loss tables**; DM shown in a **separate figure** (per-stock bars + one cross-sectional line) | MSE + QLIKE; **QLIKE is primary** |
| Corsi 2009 (JFE, original HAR) — not re-fetched | No formal DM/MCS in main tables (out-of-sample R², MSE, MAE); MCS (Hansen–Lunde–Nason 2011) postdates it | — | Point losses + R² | MSE/MAE/R² |

Conclusion: a formal predictive-accuracy test is expected in modern journal work, but the field
does **not** converge on "run DM everywhere." The canonical HARQ paper leads with **relative loss
ratios + bold + a single bootstrap test**; MCS only became routine after Hansen–Lunde–Nason (2011).

## 2. How is it MOST COMMONLY presented?

The dominant modern format (GNNHAR; Christensen et al.) is:
- **One main loss table** of point metrics, with **compact significance markers on it** —
  asterisk for the best model and a dagger for models inside the MCS. This collapses "best model"
  + "statistically indistinguishable from best" into two symbols on the table the reader already
  looks at.
- **DM statistics, when shown at all, live in a separate compact object** (a figure or a small
  side table) comparing the proposed model against **one benchmark**, not a full DM matrix in the
  body. GNNHAR: "The y-axis represents the DM test values based on QLIKE between GNNHAR2L and
  GNNHAR1L ... The horizon line represents the cross-sectional DM test value."
- HARQ's alternative, still widely copied, is **loss ratios relative to the HAR benchmark with the
  lowest ratio in bold** ("The lowest ratio in each row is highlighted in bold").

Full DM p-value matrices and multiple standalone MCS tables in the body are the exception, not the
norm, for applied volatility papers — and are rare in short conference papers.

## 3. Which loss functions — is three families (QLIKE+SE+AE) common?

No — three families is unusual, and one of ours is discouraged.

- Patton (2011) derives the class of loss functions whose **ranking is robust to noise in the
  volatility proxy**, and shows that "for almost all choices of volatility proxy most of these
  loss functions are **not robust and can lead to incorrect rankings** of volatility forecasts."
  The two standard robust members are **MSE (squared error) and QLIKE** (his Eqs. 5–6).
- **Absolute error (MAE) is among the loss functions Patton flags as non-robust** (Eq. 10, in the
  non-robust set): using it with a noisy proxy can invert the ranking. So reporting an **AE family
  as a third headline loss for volatility is not just uncommon — it is weakly justified** and can
  be attacked by a referee who knows Patton (2011).
- Field practice: report **QLIKE (primary) + MSE**. GNNHAR explicitly argues "QLIKE demonstrates
  greater statistical power than MSE in the Diebold–Mariano (DM) test. Consequently, our focus ...
  is primarily on QLIKE." HARQ reports MSE + QLIKE. Christensen et al. report MSE only.

## 4. Panel / multi-stock DM — is date-clustering the norm?

No. Multi-stock volatility papers overwhelmingly run **per-series DM and then summarize
cross-sectionally**, not a cross-sectional-dependence-robust (date-clustered) panel DM:
- Christensen et al.: "the Diebold–Mariano test ... is rejected more than 50% of the time ... across
  individual tests," i.e. **per-stock DM reported as rejection frequencies**; table entries are
  "a cross-sectional average of such pairwise relative MSEs for each stock."
- HARQ: cross-section reported as **average and median loss ratios across the individual stocks**.
- GNNHAR: per-stock DM bars plus a single **"cross-sectional DM test value"** summary line.

Our **date-clustered DM (one differential per date, robust to cross-sectional dependence) is a
rigor step BEYOND the common norm**, not a field default. It is defensible and arguably more
correct for a pooled panel, but it is an addition we are choosing, not something reviewers expect.

## 5. MCS + block-bootstrap CI together — heavier than typical

- MCS alone is now common in journal work (usually as dagger markers on the main table, per §2).
- **DM + MCS + block-bootstrap confidence intervals reported together is heavier than a typical
  conference paper.** None of the four sources read here reports all three; each picks one or two.
  HARQ uses a bootstrap Reality Check *instead of* MCS; GNNHAR/Christensen use DM + MCS without
  bootstrap loss-CIs.

---

## Recommendation

Our current apparatus — date-clustered DM × three loss families (QLIKE+SE+AE) + MCS +
block-bootstrap CIs, per horizon per panel — is **more elaborate than the field norm and includes
a loss family (AE) that Patton (2011) shows is non-robust for volatility**. It is over-built for a
SOICT-style conference paper and gives referees an easy target (the AE family). Simplify to the
GNNHAR/HARQ template:

1. **Drop the absolute-error (AE) family.** Keep **QLIKE (primary) + squared-error/MSE**. Cite
   Patton (2011) for why these two and not AE. This alone removes a third of the DM machinery and
   a citable weakness.
2. **Main results table = point metrics (QLIKE first, then RMSE/MSE) per horizon per panel, with
   two compact markers on it:** asterisk = best model, dagger = inside the 5% MCS (GNNHAR style).
   This keeps MCS but as two symbols, not standalone tables.
3. **Add ONE compact DM table (or just the asterisks):** DM statistic / p-value of each model vs
   the **HAR benchmark on QLIKE**, per horizon. One benchmark, one loss — matches GNNHAR's figure
   and HARQ's "vs HAR" framing.
4. **Keep the date-clustered DM as the single headline test** (it is defensible for the pooled
   panel and cheap to keep), but stop running per-series + clustered + MCS + bootstrap in parallel.
   State once that clustering handles cross-sectional dependence; do not also report per-series DM.
5. **Move MCS full tables and block-bootstrap CIs to an appendix, or drop the bootstrap CIs.**
   No source read here reports DM + MCS + bootstrap-CI together; it reads as over-engineering.

Net: main body shows one loss table (QLIKE+MSE, asterisk/dagger markers) + one small DM-vs-HAR
table on QLIKE, per horizon per panel. Everything else (SE/AE detail, per-series DM, bootstrap
CIs, full MCS listings) goes to an appendix or is cut. This is standard, defensible, and lighter.

## Sources

- Patton, A.J. (2011), "Volatility forecast comparison using imperfect volatility proxies,"
  *Journal of Econometrics* 160(1): 246–256. PDF:
  http://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf
- Bollerslev, T., Patton, A.J., Quaedvlieg, R. (2016), "Exploiting the errors: a simple approach
  for improved volatility forecasting," *Journal of Econometrics* 192(1): 1–18. PDF:
  http://public.econ.duke.edu/~ap172/BPQ_Exploiting_Errors_JoE_2016.pdf
- Christensen, K., Siggaard, M., Veliyev, B. (2026), "A machine learning approach to volatility
  forecasting," arXiv:2601.13014. https://arxiv.org/abs/2601.13014
- Zhang, C. et al. (2023), "GNNHAR" / graph-neural-network HAR for realized volatility,
  arXiv:2308.01419. https://arxiv.org/abs/2308.01419
- Corsi, F. (2009), "A simple approximate long-memory model of realized volatility,"
  *Journal of Financial Econometrics* 7(2): 174–196 (not re-fetched; cited from established record
  for the pre-MCS point-loss/R² presentation).
- Hansen, P.R., Lunde, A., Nason, J.M. (2011), "The Model Confidence Set," *Econometrica* 79(2)
  (origin of the MCS procedure now used as dagger markers).
- Branco, Rubesam, Zevallos, "Forecasting realized volatility: does anything beat linear models?"
  — could not be retrieved this session (not on arXiv under this title); not quoted.
