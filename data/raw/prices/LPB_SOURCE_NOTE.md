# LPB_ohlcv.csv provenance

LPB raw OHLCV was missing from the original data pipeline (only `data/processed/LPB_processed.csv`
with `parkinson_volatility` existed; no `LPB_ohlcv.csv`). Recovered 2026-08-10 from the SSI iBoard
public API (`iboard-api.ssi.com.vn/statistics/charts/history`, resolution=1D).

**Verification:** Parkinson variance recomputed from the fetched High/Low reproduces the existing
`LPB_processed.csv` `parkinson_volatility` (median |diff| < 1e-4 over ~96% of dates; residual diffs
are processed-side artifact days where H==L collapsed to 0). Confirms the correct LPB series.

**Caveat:** SSI applies a different price ADJUSTMENT convention than the other 32 tickers' raw files
(SSI levels ~1.16x vs `data/raw/prices/ACB_ohlcv.csv` recently). Because log-returns are invariant
to a constant multiplicative scale, this is immaterial for the return-GARCH baselines (differs only
on the handful of ex-dividend days). Do NOT treat LPB price LEVELS as consistent with the other
tickers' source.
