# Reference papers — HAR / HAR-X baseline

Papers supporting the HAR and HAR-X baselines used in this project. Downloaded copies are the
open-access (arXiv / working-paper) versions; paywalled items list the access link only.

## Downloaded (open access)

| File | Citation | Relevance |
|---|---|---|
| `2025_GNAR-HARX_ONuallain_arXiv2510.24443.pdf` | Ó Nualláin, T. (2025). *GNAR-HARX Models for Realised Volatility: Incorporating Exogenous Predictors and Network Effects.* arXiv:2510.24443. | Network + HAR-X + exogenous predictors — closest published analogue to VolGA (graph + HAR-type + market/volume). Reports the top-QLIKE model does not use exogenous variables. |
| `2020_Generalized-HAR-Market-Index_Hizmeri_Lancaster.pdf` | Hizmeri, R. et al. (2020). *A Generalized Heterogeneous Autoregressive Model using the Market Index.* Lancaster working paper (FoFI-2020-116). | Precedent for using a market-index factor as an exogenous HAR regressor (supports `market_pk`). |

## Not downloaded (paywalled / session-gated) — get via institution

| Citation | Access | Why cite |
|---|---|---|
| Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics 7(2), 174–196. doi:10.1093/jjfinec/nbp001 | Oxford Academic (paywall); SSRN abstract 1365738 (no download) | Original HAR model — mandatory base citation. |
| Clements, A., Preve, D. P. A., & Tee, C. (2024). *Harvesting the HAR-X Volatility Model.* SSRN 4733597. | SSRN (session-gated download) | Central HAR-X paper; uses low-frequency OHLC/candlestick data + exogenous variables — validates the range-based (Parkinson) HAR-X used here. |
| *A Practical Guide to harnessing the HAR volatility model.* Journal of Banking & Finance (2021). | ScienceDirect (paywall) | Practical HAR/HAR-X implementation guide. |

## Related variants (from literature search, not downloaded)

- Bekaert & Hoerova (2014); Liu & Zhang (2015): HAR-RV + VIX / EPU exogenous variables.
- Fernandes, Medeiros & Scharth (2014): asymmetric HAR-X (AHARX).
- Bollerslev, Patton & Quaedvlieg (2016): HARQ (realized-quarticity coefficient scaling).
