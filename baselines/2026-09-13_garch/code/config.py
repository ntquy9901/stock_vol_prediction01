"""Single source of truth for this baseline's tunable constants (avoids the config-hardcode gate).

GARCH(1,1) / GJR-GARCH(1,1,1) conditional-variance benchmark. Constants below define the estimation
scale, the innovation model, the fold gate, and the fallback threshold. See design/design.md.
"""
SCALE = 100.0                # returns are scaled x100 for arch's numerical conditioning; forecasts /SCALE**2
MEAN = "Constant"            # arch mean model (epsilon = r - mu)
DIST = "normal"              # innovation distribution (symmetric -> P(eps<0)=0.5 in the GJR multistep)
HORIZONS = (1, 5, 10, 22)    # forecast horizons (match the sibling comparison table)
MIN_ROWS = {"sp500": 30000, "default": 3000}   # min pooled train rows per fold (mirrors sibling fold gate)
MIN_TRAIN_OBS = 250          # min per-ticker train returns to attempt an ML fit (~1 trading year), else fallback
PERSIST_LO = 0.0             # reversion persistence must satisfy PERSIST_LO < phi < PERSIST_HI (else fallback)
PERSIST_HI = 1.0
VAR_RATIO_CAP = 1000.0       # the fit-implied unconditional variance AND every forecast must stay within this
                             # factor of the ticker's OWN sample return variance; else the fit is degenerate
                             # (-> fallback) / the forecast is clipped. Guards near-IGARCH fits where arch
                             # lands on omega~0 (variance collapses to ~1e-9) or phi~1 (unconditional variance
                             # explodes) — orders of magnitude off the data — which otherwise dominate the
                             # pooled QLIKE (root cause of the SP500 blow-up). 1000x = 3 orders of magnitude,
                             # loose enough that a legitimate high-persistence fit (uncond ~ a few x sample
                             # variance) is untouched, tight enough to reject the 1e5-1e8x pathologies.
