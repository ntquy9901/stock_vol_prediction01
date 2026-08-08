# HAR as a Benchmark for Daily Realized/Parkinson Volatility Forecasting: Literature Review

Date: 2026-08-08
Scope: Two questions for daily realized/Parkinson volatility forecasting — (1) which models have actually
been shown to beat HAR out-of-sample, and under what conditions; (2) how the literature benchmarks against
HAR (baseline suite + statistical tests). Anchored to this project's setting: VN30 (33 Vietnamese stocks),
daily Parkinson volatility, 1/5/10/22-day-ahead.

Citation policy: every reference below was checked against a publisher, arXiv, RePEc, or SSRN record during
this review. Items whose exact venue/author list could not be fully confirmed are marked `[partially verified]`
or `[unverified]` and are not relied on for load-bearing claims.

---

## 1. Is HAR genuinely hard to beat?

The consensus in the realized-volatility literature is that the HAR model is a strong, parsimonious benchmark
that many more complex models fail to beat out-of-sample, especially on squared-error losses and at short
horizons.

- **Corsi (2009)** introduced the HAR-RV model: an additive cascade of daily/weekly/monthly realized-volatility
  lags estimated by OLS. Despite not being a true long-memory process, it reproduces long memory, fat tails,
  and self-similarity, and delivers strong out-of-sample forecasts. It has become the field's default benchmark.
  - Fulvio Corsi (2009), "A Simple Approximate Long-Memory Model of Realized Volatility," *Journal of Financial
    Econometrics* 7(2): 174–196. DOI 10.1093/jjfinec/nbp001. Verified (Oxford Academic).

- **Audrino & Chassot (2024), "HARd to Beat"** is the most direct recent statement of the consensus. They show
  that machine-learning models fail to surpass HAR once HAR is fitted with a properly specified rolling-window /
  re-estimation scheme. The headline is a methodological warning: many reported "ML beats HAR" wins are partly
  an artifact of using a weak/misspecified HAR fitting scheme (training-window length and re-estimation
  frequency materially change HAR's measured accuracy).
  - Francesco Audrino & Jonathan Chassot (2024), "HARd to Beat: The Overlooked Impact of Rolling Windows in the
    Era of Machine Learning," arXiv:2406.08041. Verified (arXiv).

- **Christensen, Siggaard & Veliyev (2023)** is the most-cited recent counterweight: with minimal tuning, ML
  (regularization, trees, neural nets) beats the HAR lineage on Dow Jones constituents — but the gains are
  concentrated at longer horizons and when the information set is enriched; at h=1 with only daily/weekly/monthly
  RV lags the edge is modest (up to ~11.8% MSE reduction on some stocks). This is the paper most often cited both
  for and against "ML beats HAR," which itself indicates the effect is real but conditional.
  - Kim Christensen, Mathias Siggaard & Bezirgen Veliyev (2023), "A Machine Learning Approach to Volatility
    Forecasting," *Journal of Financial Econometrics* 21(5): 1680–1727. DOI 10.1093/jjfinec/nbac020. Verified.

- **Foundation-model comparison (2026), "Forecasting Realized Volatility with Time Series Foundation Models"**
  runs nine zero-shot time-series foundation models against eight econometric specs (including the HAR family)
  across 50 assets and three horizons. Foundation models do not deliver a uniform gain: pooled losses favor them,
  but the advantage concentrates in a few outlier assets; only one small model (Tiny Time Mixers) beats the
  benchmark at every horizon, by a narrow margin, while the others do not improve on Log-HAR. The econometric
  benchmarks remain competitive throughout.
  - arXiv:2607.05291 (2026), "Forecasting Realized Volatility with Time Series Foundation Models: A Comparison
    with Econometric Benchmarks." Verified (arXiv listing); exact author list `[partially verified]`.

- The 2025 review in *Financial Innovation* frames the same tension: HAR "has established itself as a widely
  adopted benchmark ... consistently demonstrating strong forecasting accuracy and often outperforming more
  complex models," while noting a top hybrid CNN-LSTM as the best-performing architecture in its survey. The
  review explicitly states the literature does **not** reach a unanimous "ML beats HAR" verdict.
  - "Advances in Forecasting Realized Volatility: A Review of Methodologies," *Financial Innovation* (2025).
    DOI 10.1186/s40854-025-00809-5. Verified (Springer, open access).

**Takeaway.** HAR is a genuinely strong benchmark. Reported wins over it are real but conditional on (a) the
loss function (QLIKE more often than MSE/RMSE), (b) longer horizons, (c) an enriched information set beyond RV
lags, and (d) a fairly-fitted HAR. On plain RMSE at short horizons with an RV-only information set, HAR is
routinely not beaten.

---

## 2. Models reported to beat HAR — with specifics and caveats

The table in Section 6 consolidates this. Narrative detail and caveats per family:

- **HAR-family extensions (HARQ, HAR-J/HAR-CJ, SHAR, CHAR, log-HAR).** These are the most reliable "beats plain
  HAR" results because they are apples-to-apples (same estimator class, same loss) and are significance-tested.
  Details in Section 3.

- **Realized GARCH (Hansen, Huang & Shek 2012).** A joint model of returns and a realized measure via a
  measurement equation; documented "substantial improvements in empirical fit over standard GARCH." Note the
  comparison is primarily against daily-data GARCH, not always against HAR head-to-head; and gains are often
  in-sample fit / density forecasting rather than point-RV RMSE. Verified: *Journal of Applied Econometrics*
  27(6): 877–906, DOI 10.1002/jae.1234.

- **Neural networks — LSTM/RNN (Bucci 2020).** Feed-forward and recurrent nets (LSTM, NARX) reported to
  outperform ARFIMA/traditional econometric models on S&P 500 monthly RV, on MSE and QLIKE, with the edge
  largest in high-volatility periods. Caveat: single index (S&P 500), and the comparison set is econometric
  (ARFIMA-type) more than a fully-tuned HAR suite. Verified: *Journal of Financial Econometrics* 18(3): 502–531,
  DOI 10.1093/jjfinec/nbaa008.

- **LSTM with rich features — Rahimikia & Poon (2020).** LSTM fed HAR variables + limit-order-book (LOB) +
  news sentiment beats HAR in ~90% of the out-of-sample period **except during extreme volatility**, on 23
  NASDAQ stocks. Critically, their own HAR-family investigation found **CHAR** to be the best HAR variant — so
  their "beat" is partly against a strengthened HAR, and the win depends on adding LOB/news features (not RV
  alone). Verified: SSRN 3707796 (ML paper) and SSRN 3684040 (HAR+LOB+news). Peer-reviewed venue `[unverified]`.

- **DeepVol — dilated causal convolutions (Moreno-Pino et al. 2024).** Works from raw intraday high-frequency
  data (bypassing pre-computed realized measures) and outperforms HAR, LSTM, and MLP on NASDAQ-100 day-ahead
  volatility, remaining robust in high-vol regimes. The RV-input variant (DeepVol-RM) also beats HAR, but the
  authors attribute this partly to learned weighting of the 22-day window vs HAR's fixed aggregation. Requires
  intraday data — not applicable to a daily-OHLCV Parkinson setup without HF data. Verified: *Quantitative
  Finance* (2024), DOI 10.1080/14697688.2024.2387222; arXiv:2210.04797. Second author name `[partially verified]`.

- **HARNet — CNN with HAR-inspired dilations (Reisenhofer, Bayer & Hautsch 2022).** A convolutional net that
  generalizes HAR's fixed lag weights; reported improvements over HAR. Uses pre-computed realized measures.
  Verified as a paper; exact peer-reviewed venue `[partially verified]` (widely cited as a 2022 working paper /
  *Econometrics and Statistics*-track).

- **GNN / graph-transformer — Chen & Robert (2022).** A graph transformer over ~500 S&P 500 stocks using LOB +
  relational data reports better performance than "other benchmarks" for short-term multivariate RV. Caveat:
  the abstract does not explicitly name HAR as the beaten benchmark, and follow-up papers flag limitations in
  interpretability and benchmarking rigor. Requires intraday/LOB + cross-sectional relations. Verified:
  Proc. 3rd ACM ICAIF (2022), pp. 156–164, DOI 10.1145/3533271.3561663; arXiv:2112.09015.

- **Foundation / TimesFM (Goel, Pasricha, Magris & Kanniainen 2025).** Zero-shot TimesFM is only a reasonable
  baseline; **incremental fine-tuning is essential** to be competitive with econometric benchmarks. This, plus
  the 2026 comparison paper above, indicates foundation models do not yet reliably beat HAR out-of-the-box.
  Verified: arXiv:2505.11163.

- **Tree ensembles (XGBoost / LightGBM / RF).** A comparative study reports Boosting achieving the lowest QLIKE
  (0.1219) vs HAR-RV (0.1482), with Diebold–Mariano confirming significance, and LSTM being the *worst* ML model
  (beaten by HAR-RV on MAE/RMSE). This is a recurring pattern: gradient-boosted trees, not deep nets, are the ML
  family most often reported to beat HAR on QLIKE. Source: comparative MSc/working study `[unverified peer-review]`
  (DiVA, diva2:2031701) — used only illustratively, not as a load-bearing citation.

**Explicit caveat on "beats HAR" claims.** A large share are QLIKE-only, single-market (usually S&P 500 or DJIA
constituents), depend on HF/LOB/news features unavailable in a daily-OHLCV setup, are at longer horizons, or are
not significance-tested. He (2023) [SSRN, `[unverified peer-review]`] reports LSTM only 4–6% better than HAR-RV
on the S&P 500 and **not statistically significant**; Kilic (2025) [`[unverified]`] reports HAR-RV consistently
beating LSTM/XGBoost for one-day-ahead forecasts when the information set is limited. These match this project's
own finding.

---

## 3. HAR variants as stronger baselines

Plain HAR is often not the strongest member of its own family. A referee-credible volatility paper is usually
expected to benchmark against at least one strengthened HAR, or justify using plain HAR. The main variants:

- **HARQ (Bollerslev, Patton & Quaedvlieg 2016).** Lets the autoregressive coefficients vary with the estimated
  measurement error (realized quarticity): when RV is a precise signal, persistence is higher and forecasts more
  responsive. HARQ "outperforms the forecasts from several existing benchmark models" on S&P 500 and DJIA
  constituents. This is the single most cited "stronger-than-HAR baseline." Verified: *Journal of Econometrics*
  192(1): 1–18, DOI 10.1016/j.jeconom.2015.10.007.

- **HAR-J / HAR-CJ / HAR-RV-CJ (Andersen, Bollerslev & Diebold 2007).** Decompose RV into a continuous component
  and a jump component (via bipower variation), each with its own dynamics. The jump component is less persistent;
  separating it yields significant out-of-sample gains. Verified: *Review of Economics and Statistics* 89(4):
  701–720, DOI 10.1162/rest.89.4.701.

- **SHAR — semivariance HAR (Patton & Sheppard 2015).** Splits the first RV lag into positive and negative
  realized semivariances ("good"/"bad" volatility). Future volatility loads far more on past negative-return
  variation (leverage). DM tests reject equal performance in favor of the semivariance model for 20–30% of series
  and are uniformly positive for the S&P 500 ETF. Verified: *Review of Economics and Statistics* 97: 683–697
  (2015); issue number `[partially verified]` (listed as 97(2) on RePEc).

- **CHAR (continuous-component HAR).** Uses the continuous (bipower-variation) component in place of RV as the
  regressor. Notably, Rahimikia & Poon (2020) found CHAR the **best-performing HAR variant** across 23 NASDAQ
  stocks over a long out-of-sample window. CHAR appears as a standard benchmark in Bollerslev–Patton–Quaedvlieg
  (2016); its lineage traces to the continuous/jump decomposition of Andersen–Bollerslev–Diebold (2007). Origin
  as a distinctly named "CHAR" model `[partially verified]` — attribute the idea to the continuous-component
  literature rather than to a single canonical "CHAR" paper.

- **log-HAR.** HAR estimated on log-RV (variance-stabilizing; forecasts are exp-transformed back). Frequently the
  hardest econometric baseline to beat in recent DL comparisons — e.g., the 2026 foundation-model study found most
  foundation models "do not improve on Log-HAR." A cheap, strong baseline to include.

- **HARX / HAR with exogenous regressors.** HAR augmented with exogenous predictors (implied volatility, LOB
  variables, macro announcements, news sentiment — e.g., Rahimikia & Poon's "CHARx"). For an emerging-market
  daily setup, HARX with a news/sentiment regressor is the most natural way to inject the project's news signal
  into a strong linear baseline (see Section 5).

Relevance to this project: the project currently benchmarks deep models against **plain HAR**. Adding **HARQ**
and **log-HAR** (both computable from the same realized-measure inputs, no extra data) would materially
strengthen the baseline and preempt the standard reviewer objection that the deep model only beats a weak HAR.

---

## 4. Standard baseline suite + benchmarking methodology

What a volatility-forecasting paper is generally expected to include:

**Baseline suite (comparison models):**
- **Random walk / persistence** (yesterday's RV) and/or a naive historical-mean forecast — the trivial floor.
- **EWMA / RiskMetrics** — exponentially weighted variance.
- **GARCH family** — at minimum GARCH(1,1); ideally an asymmetric member (GJR-GARCH or EGARCH) to capture the
  leverage effect; and **Realized GARCH** (Hansen, Huang & Shek 2012) when a realized measure is available.
- **HAR family** — plain HAR (Corsi 2009) plus at least one strengthened variant (HARQ, log-HAR, HAR-CJ, or SHAR).
- The proposed model(s).

**Evaluation metrics and tests:**
- **Robust loss functions: QLIKE and MSE.** Patton (2011) proves that among common volatility loss functions,
  only MSE and QLIKE give rankings robust to noise in the (imperfect) volatility proxy. Reporting both is the
  standard defensible choice; QLIKE additionally has higher statistical power to discriminate models. Verified:
  *Journal of Econometrics* 160(1): 246–256, DOI 10.1016/j.jeconom.2010.03.034.
- **Diebold–Mariano test (1995)** for pairwise equal-predictive-accuracy testing under a chosen loss; works with
  non-quadratic/asymmetric losses. Verified: *Journal of Business & Economic Statistics* 13(3): 253–263,
  DOI 10.1080/07350015.1995.10524599. (Note Diebold's own caution: the DM test is for comparing *forecasts*, not
  nested *models*; use with care for nested HAR variants.)
- **Model Confidence Set (Hansen, Lunde & Nason 2011)** to identify, at a confidence level, the set of models not
  statistically distinguishable from the best — the multi-model analogue of DM. Use the range statistic per the
  authors' corrigendum. Verified: *Econometrica* 79(2): 453–497, DOI 10.3982/ECTA5771.
- **Mincer–Zarnowitz regression** (regress realized on forecast; test intercept=0, slope=1) for forecast
  unbiasedness/efficiency, and **out-of-sample R²** (a.k.a. MZ-R²) for variance explained. (Foundational MZ
  reference is Mincer & Zarnowitz 1969, NBER — `[not separately re-verified in this review]`; it is a standard
  textbook diagnostic.)

**Proxy note for this project.** Parkinson (range-based) volatility is itself a noisy proxy for integrated
variance. Patton (2011)'s robustness result is exactly why QLIKE + MSE (rather than MAE, R² alone, or
direction) should be the primary evaluation losses when the target is a noisy proxy — this directly supports
this project's use of QLIKE as a primary criterion.

---

## 5. Recommendations for this project's paper

1. **Strengthen the HAR baseline.** Plain HAR alone invites the reviewer objection "you only beat a weak HAR."
   Add **log-HAR** and **HARQ** — both computable from the same realized-measure inputs, no new data. If a jump
   or semivariance decomposition is feasible from the OHLC/Parkinson pipeline, HAR-CJ / SHAR would further
   harden the baseline. These are cheap and directly address the most likely critique.

2. **Add at least one GARCH-family and one naive baseline.** GJR-GARCH or EGARCH (asymmetry) plus random-walk/
   persistence and EWMA complete the expected suite. Realized GARCH is optional but well-regarded if a realized
   measure is modeled jointly with returns.

3. **The "deep beats HAR only on QLIKE, ties on RMSE, direction ~random" story is consistent with the
   literature — frame it as a finding, not a failure.** Supporting evidence:
   - QLIKE-concentrated wins are the norm: He (2023), the tree-ensemble comparisons, and Bucci (2020) all show
     the DL/ML edge appears on QLIKE (asymmetric, penalizes under-prediction of high vol) more than on
     symmetric RMSE.
   - "HAR ties/wins on RMSE" matches Audrino & Chassot (2024), Kilic (2025), the COMEX-copper study, and the
     emerging-market (South Africa) study where HAR gives the better out-of-sample forecast.
   - Near-random direction on a mean-reverting/anti-persistent daily target is expected: volatility forecasting
     is a level/magnitude task; directional accuracy is not a standard headline metric in this literature and
     should be reported as secondary, with the anti-persistence autocorrelation noted as the structural cause.

4. **Report QLIKE + MSE as primary, with DM and (ideally) MCS tests.** This is the methodological bar. Given the
   project already runs multi-seed experiments, pairing per-model losses with DM/MCS across the baseline suite is
   the highest-leverage addition for credibility. Cite Patton (2011) for the loss choice, DM (1995) and MCS
   (2011) for the tests.

5. **What it would actually take to beat HAR on RMSE for a daily emerging-market panel.** The literature points
   to three levers, in rough order of plausibility for a daily-OHLCV VN30 setup:
   - **Exogenous information, especially news/sentiment in high-volatility regimes** (HARX/CHARx-style).
     Rahimikia & Poon (2020) get their HAR-beating edge precisely from LOB + news features, and it *disappears
     in extreme-volatility periods* — the opposite of where news might be expected to help, which is a useful
     nuance. Implied volatility gave a ~10% forecast improvement in the emerging-market (South Africa) study.
     For VN30, a news signal is most likely to pay off as a regime-conditional exogenous regressor, not as a
     blanket deep-model input.
   - **Measurement-error correction (HARQ-style).** Emerging-market daily proxies are noisier; the HARQ
     mechanism (down-weighting noisy RV days) is designed exactly for this and is cheap to add.
   - **Realized GARCH / joint return-volatility modeling** if a realized measure and returns are modeled together.
   - Pure deep sequence models (LSTM/GNN/foundation) *without* extra information or HF data are, per the
     consensus, unlikely to beat a well-fitted HAR on RMSE at short horizons — consistent with this project's
     own results.

6. **Novelty framing / gap.** No genuine *multi-market emerging-panel* HAR-vs-DL study with news was located in
   this review (closest work is single-country South Africa or cross-market comparisons). A rigorous VN30 panel
   study — HAR/HARQ/log-HAR + GARCH baselines, QLIKE/MSE + DM/MCS, honest reporting that the deep/news edge is
   QLIKE-concentrated and regime-dependent — occupies a defensible, under-served niche.

---

## 6. Comparison table

Legend: "Beats HAR on?" = the metric(s) where the model is reported to beat HAR (or the HAR family). "Sig-tested"
= whether the win was tested (DM / MCS) in the cited source. All citations verified unless marked.

| Model / family | Beats HAR? | Metric of win | Market / horizon | Sig-tested | Citation |
|---|---|---|---|---|---|
| HARQ (measurement-error-corrected HAR) | Yes (vs plain HAR) | MSE/QLIKE | S&P500 + DJIA constituents; h=1,5,22 | Yes (MCS/DM) | Bollerslev, Patton & Quaedvlieg 2016, *J. Econometrics* 192:1–18 |
| HAR-J / HAR-CJ (jump decomposition) | Yes (vs plain HAR) | MSE-type OOS | FX, S&P500, T-bonds; h=1,5,22 | Yes | Andersen, Bollerslev & Diebold 2007, *REStat* 89:701–720 |
| SHAR (semivariance HAR) | Yes (vs plain HAR) | MSE/QLIKE | S&P500 ETF + 105 stocks; h=1 | Yes (DM; 20–30% of series) | Patton & Sheppard 2015, *REStat* 97:683–697 |
| CHAR (continuous-component HAR) | Yes (best HAR variant) | RMSE/QLIKE | 23 NASDAQ stocks; h=1 | Partly | Rahimikia & Poon 2020, SSRN 3684040 |
| log-HAR | Baseline often unbeaten | MSE/QLIKE | 50 assets; h=1,5,22 | Yes | arXiv:2607.05291 (2026) `[partially verified]` |
| Realized GARCH | Yes (vs daily GARCH) | In-sample fit / density | DJIA + ETF | Partly | Hansen, Huang & Shek 2012, *JAE* 27:877–906 |
| LSTM/RNN (RV-only) | Mixed → often No | QLIKE (when it wins) | S&P500 index | Sometimes | Bucci 2020, *JFEC* 18:502–531 (wins); He 2023 SSRN `[unverified]` (no sig. win) |
| LSTM + LOB + news | Yes (≈90% of OOS, not in extreme vol) | RMSE/QLIKE | 23 NASDAQ stocks; h=1 | Partly | Rahimikia & Poon 2020, SSRN 3707796 |
| DeepVol (dilated causal CNN, raw HF) | Yes | MSE/QLIKE/MedAE | NASDAQ-100; h=1 | Partly | Moreno-Pino et al. 2024, *Quant. Finance*; arXiv:2210.04797 |
| HARNet (HAR-inspired CNN) | Yes | MSE-type | equity indices; h=1 | Partly | Reisenhofer, Bayer & Hautsch 2022 `[partially verified]` |
| GNN / graph transformer | Yes (vs "benchmarks") | RMSE-type (LOB) | ~500 S&P500 stocks; intraday | Weak / unclear | Chen & Robert 2022, ACM ICAIF; arXiv:2112.09015 |
| Tree ensembles (XGBoost/LightGBM/RF) | Yes | QLIKE | S&P500 RV | Yes (DM) | Comparative study, DiVA diva2:2031701 `[unverified peer-review]` |
| Foundation models (TimesFM / TSFMs) | Mostly No zero-shot; marginal fine-tuned | pooled loss (few outlier assets) | 50 assets; h=1,5,22 | Yes | arXiv:2505.11163 (2025); arXiv:2607.05291 (2026) `[partially verified]` |
| ML (regularization/trees/NN), enriched info | Yes (esp. longer h) | MSE | DJIA constituents; h=1..longer | Yes | Christensen, Siggaard & Veliyev 2023, *JFEC* 21:1680–1727 |
| ML generally, RV-only, well-fitted HAR | No | RMSE/MSE | S&P500, COMEX, S.Africa | Yes (DM) | Audrino & Chassot 2024, arXiv:2406.08041; Kilic 2025 `[unverified]` |

---

## Verified reference list (canonical, load-bearing)

- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. *Journal of Financial
  Econometrics* 7(2): 174–196. DOI 10.1093/jjfinec/nbp001.
- Bollerslev, T., Patton, A. J., & Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved
  volatility forecasting. *Journal of Econometrics* 192(1): 1–18. DOI 10.1016/j.jeconom.2015.10.007.
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of
  Econometrics* 160(1): 246–256. DOI 10.1016/j.jeconom.2010.03.034.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic
  Statistics* 13(3): 253–263. DOI 10.1080/07350015.1995.10524599.
- Hansen, P. R., Lunde, A., & Nason, J. M. (2011). The Model Confidence Set. *Econometrica* 79(2): 453–497.
  DOI 10.3982/ECTA5771.
- Andersen, T. G., Bollerslev, T., & Diebold, F. X. (2007). Roughing It Up: Including Jump Components in the
  Measurement, Modeling, and Forecasting of Return Volatility. *Review of Economics and Statistics* 89(4):
  701–720. DOI 10.1162/rest.89.4.701.
- Patton, A. J., & Sheppard, K. (2015). Good Volatility, Bad Volatility: Signed Jumps and the Persistence of
  Volatility. *Review of Economics and Statistics* 97: 683–697.
- Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH: a joint model for returns and realized measures
  of volatility. *Journal of Applied Econometrics* 27(6): 877–906. DOI 10.1002/jae.1234.
- Christensen, K., Siggaard, M. V., & Veliyev, B. (2023). A Machine Learning Approach to Volatility Forecasting.
  *Journal of Financial Econometrics* 21(5): 1680–1727. DOI 10.1093/jjfinec/nbac020.
- Bucci, A. (2020). Realized Volatility Forecasting with Neural Networks. *Journal of Financial Econometrics*
  18(3): 502–531. DOI 10.1093/jjfinec/nbaa008.
- Audrino, F., & Chassot, J. (2024). HARd to Beat: The Overlooked Impact of Rolling Windows in the Era of Machine
  Learning. arXiv:2406.08041.
- Moreno-Pino, F., et al. (2024). DeepVol: Volatility Forecasting from High-Frequency Data with Dilated Causal
  Convolutions. *Quantitative Finance*. DOI 10.1080/14697688.2024.2387222; arXiv:2210.04797.
- Chen, Q., & Robert, C.-Y. (2022). Multivariate Realized Volatility Forecasting with Graph Neural Network.
  *Proc. 3rd ACM ICAIF*, 156–164. DOI 10.1145/3533271.3561663; arXiv:2112.09015.
- Rahimikia, E., & Poon, S.-H. (2020). Machine Learning for Realised Volatility Forecasting (SSRN 3707796); and
  Big-data HAR + LOB + news (SSRN 3684040).
- Goel, A., Pasricha, P., Magris, M., & Kanniainen, J. (2025). Foundation Time-Series AI Model for Realized
  Volatility Forecasting. arXiv:2505.11163.
- "Advances in Forecasting Realized Volatility: A Review of Methodologies" (2025). *Financial Innovation*.
  DOI 10.1186/s40854-025-00809-5.

### Items used with caveats (not load-bearing)
- arXiv:2607.05291 (2026), "Forecasting Realized Volatility with Time Series Foundation Models: A Comparison with
  Econometric Benchmarks" — venue/authors `[partially verified]`.
- Reisenhofer, Bayer & Hautsch (2022), HARNet — exact peer-reviewed venue `[partially verified]`.
- He (2023), "Forecasting Stock Market Volatility: A Comparison of GARCH, HAR-RV, and LSTM Models" — SSRN,
  peer-review `[unverified]`.
- Kilic (2025); Souto & Moradi (2023); Li et al. (2025) — cited via secondary sources; full bibliographic detail
  `[unverified]`.
- "Improving realised volatility forecast for emerging markets," *Journal of Economics and Finance* (2024),
  DOI 10.1007/s12197-024-09701-x — South Africa; HAR better OOS, implied vol adds ~10%. `[partially verified]`.
- Comparative ML study, DiVA diva2:2031701 (tree-ensemble QLIKE result) — `[unverified peer-review]`.
- Mincer & Zarnowitz (1969), NBER — standard MZ-regression diagnostic, `[not separately re-verified here]`.
