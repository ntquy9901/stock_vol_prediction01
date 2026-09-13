# Data-Driven, Causal Selection of HAR Lag Windows

> Paper-ready methodology section for the principled HAR-family feature set
> (see spec `docs/superpowers/specs/2026-09-13-principled-har-features-design.md`).
> Verify page/volume numbers against the original sources before submission.

## 1. Motivation: long memory in volatility and the HAR model

Realized volatility exhibits **long memory**: its autocorrelation function (ACF) decays hyperbolically (slowly)
rather than at the geometric rate of a finite-order AR(p) process (Andersen, Bollerslev, Diebold & Labys, 2003;
Comte & Renault, 1998). This is formally captured by the ARFIMA(p, d, q) class with a fractional differencing
parameter d ∈ (0, 0.5).

Corsi (2009) introduced the **HAR-RV** model as a *parsimonious approximation* of long memory: rather than
estimating a fractional d, it linearly combines a small number of volatility averages over distinct **time
horizons** (daily / weekly / monthly), motivated by the *heterogeneous market hypothesis* — investor groups
operating at different trading frequencies generate a multi-scale dependence structure. In general form,

$$\log RV_{t+h} = \beta_0 + \sum_{k=1}^{K} \beta_k\, \overline{\log RV}^{(w_k)}_{t} + \varepsilon_{t+h}, \qquad
\overline{\log RV}^{(w)}_t = \frac{1}{w}\sum_{i=0}^{w-1} \log RV_{t-i}.$$

Classical HAR fixes K = 3 and $(w_1, w_2, w_3) = (1, 5, 22)$ ("day/week/month"). **The problem:** these windows
are calendar conventions, not estimated from the data, and may be suboptimal for a specific market (e.g., a
frontier market such as HOSE).

## 2. Diagnosing long memory: the GPH estimator

Before selecting windows, we **verify** that a multi-scale structure is warranted by estimating the fractional
parameter d of $\log RV$ via the log-periodogram regression of **Geweke & Porter-Hudak (1983, GPH)**. With the
periodogram $I(\lambda_j)$ at the low Fourier frequencies $\lambda_j = 2\pi j/n,\ j = 1,\dots,m$,

$$\ln I(\lambda_j) = c - d\cdot \ln\!\Big(4\sin^2\tfrac{\lambda_j}{2}\Big) + \varepsilon_j,$$

so $\hat{d}$ equals the **negative of the OLS slope**. We use the bandwidth $m = \lfloor n^{0.5} \rfloor$ (a
standard default; see Hurvich, Deo & Brodsky, 1998, on the bias–variance trade-off in m).

Interpretation: $\hat{d} \approx 0$ implies no long memory (a single lag suffices); $\hat{d}$ in the range
≈ 0.3–0.5 indicates **strong long memory**, so multiple horizons — and hence a multi-window HAR — are justified.
This serves as **supporting evidence**, not as a forecast. ACF/PACF inspection is used as a complementary visual
check.

## 3. Window selection by AIC over a candidate grid

Let the candidate grid be $\mathcal{G} = \{3, 5, 10, 22, 44, 66\}$ days. For each triple
$(w_s, w_m, w_l) \subset \mathcal{G}$, we fit a HAR-OLS of $\log RV_t$ on the **lag-one** trailing means (past
information only ⇒ causal) of $\log RV$ over the three windows, and compute the **Akaike Information Criterion**
(Akaike, 1974):

$$\text{AIC} = n\ln\!\Big(\tfrac{\text{RSS}}{n}\Big) + 2k,$$

where k is the number of parameters. The selected windows minimise AIC. AIC balances goodness-of-fit against
parsimony, preventing the mechanical preference for ever-longer windows that a raw fit criterion would induce.
This is a **standard model-selection rule**, in contrast to a manual choice of windows.

## 4. Why the procedure must be causal, one-shot, and fixed across folds

Under the pseudo-out-of-sample (walk-forward) evaluation principle, **any modelling choice — including feature
windows — that uses test-period data inflates measured performance** (look-ahead / data-snooping). Accordingly:

- The windows are selected **only from the training data preceding the first test fold** ($\text{date} <
  \text{fold}_1$), pooled across tickers.
- They are chosen **once** and **held fixed across all folds**, so feature semantics are invariant across folds
  (features never "change meaning" fold-to-fold, and the comparison stays fair).
- The final performance verdict uses the **Diebold–Mariano test** (Diebold & Mariano, 1995) on date-clustered
  losses — AIC only *selects* the windows; DM *adjudicates* out-of-sample.

## 5. Illustration

On the HOSE training sample (Parkinson log-RV, 2015 to the first fold boundary): suppose $\hat{d} \approx 0.45$,
confirming long memory; scanning the $\binom{6}{3} = 20$ triples, the minimum AIC falls at $(5, 22, 66)$, which
is then used. If $(1, 5, 22)$ ranks near the top, we conclude that "Corsi's windows are confirmed by the HOSE
data"; otherwise we adopt the data-selected windows, with the AIC as evidence.

## 6. Limitations

- **GPH** bias and variance are sensitive to the bandwidth m; being semiparametric, $\hat{d}$ is diagnostic
  only. (Local Whittle estimation — Robinson, 1995 — is a more robust alternative if needed.)
- **In-sample AIC** selection does not guarantee out-of-sample optimality; hence the downstream DM
  non-inferiority test is the true arbiter.
- **Daily OHLC only** precludes intraday diagnostics (realized quarticity, bipower variation / jumps).
- **Thin markets (HOSE):** short history and many near-zero volatility days make both the GPH estimate and the
  AIC grid noisier.

## References

- Akaike, H. (1974). *A New Look at the Statistical Model Identification.* IEEE Transactions on Automatic
  Control, 19(6), 716–723.
- Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). *Modeling and Forecasting Realized
  Volatility.* Econometrica, 71(2), 579–625.
- Comte, F., & Renault, E. (1998). *Long Memory in Continuous-Time Stochastic Volatility Models.* Mathematical
  Finance, 8(4), 291–323.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial
  Econometrics, 7(2), 174–196.
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing Predictive Accuracy.* Journal of Business & Economic
  Statistics, 13(3), 253–263.
- Geweke, J., & Porter-Hudak, S. (1983). *The Estimation and Application of Long Memory Time Series Models.*
  Journal of Time Series Analysis, 4(4), 221–238.
- Hurvich, C. M., Deo, R., & Brodsky, J. (1998). *The Mean Squared Error of Geweke and Porter-Hudak's Estimator
  of the Memory Parameter of a Long-Memory Time Series.* Journal of Time Series Analysis, 19(1), 19–46.
- Parkinson, M. (1980). *The Extreme Value Method for Estimating the Variance of the Rate of Return.* Journal of
  Business, 53(1), 61–65.
- Patton, A. J., & Sheppard, K. (2015). *Good Volatility, Bad Volatility: Signed Jumps and the Persistence of
  Volatility.* Review of Economics and Statistics, 97(3), 683–697.
- Robinson, P. M. (1995). *Gaussian Semiparametric Estimation of Long Range Dependence.* Annals of Statistics,
  23(5), 1630–1661.
