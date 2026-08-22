# Output parameterization and QLIKE seed stability — robustness study

Date: 2026-08-22. Scope: the deep model's forecast target parameterization and its effect on QLIKE
seed stability, on the masked-rich panel (LSTM without graph). Not a change to the submitted paper's
numbers; a methods/robustness finding produced during review.

## Motivation
On the single-seed S&P 500 quick check, the deep model's QLIKE was large and seed-sensitive (e.g. h5
LSTM QLIKE 1.04 at seed 42 vs 0.47 at seed 123). Diagnosis: the additive standardized-target design
(`(y-mu)/sigma`, denormalized linearly, then a post-hoc floor `1e-2*mu`) lets a forecast fall to the
floor for a node that later spikes, so `y/p` and hence QLIKE explode on a few observations; which nodes
collapse depends on the seed.

## Configurations (LSTM, no graph)
- **A** z-score + linear + floor (current): target `(y-mu)/sigma`; infer `max(pn*sigma+mu, 1e-2*mu)`.
- **B** ratio + linear + floor: target `y/mu`; infer `max(pn*mu, 1e-2*mu)`.
- **C** ratio + exp (no floor): target `y/mu`; infer `exp(pn)*mu`.
- **D** ratio + softplus (no floor): target `y/mu`; infer `softplus(pn)*mu`.

The ratio target `r = y/mu_i` makes the training loss a node-scaled MSE, equivalent to weighting the
raw-scale squared error by `1/mu_i^2`. A machine-epsilon (`np.finfo(float32).tiny`) is applied to every
forecast as a numerical safeguard for the QLIKE division/log — distinct from the economic `1e-2*mu`
floor. QLIKE clamps target and forecast to a shared `1e-8` (existing protocol; `y==0` Parkinson-variance
days are clamped consistently across configs). QLIKE protects the forecast side only; the target-side
`y==0` handling is the shared 1e-8 clamp.

## Results

### S&P 500 h5 (single-cell diagnosis, 2 seeds; 442 nodes, 717,772 obs)
| Config | QLIKE mean | QLIKE spread | R^2 |
|---|---|---|---|
| A z-score+linear+floor | 0.7585 | 0.5696 | 0.131 |
| B ratio+linear+floor | 0.5061 | 0.0395 | 0.102 |
| C ratio+exp | 0.4953 | 0.0014 | 0.101 |
| D ratio+softplus | 0.4863 | 0.0003 | 0.095 |

Note: MSE differences are real but ~1e-7 in magnitude; the R^2 column (common denominator across configs)
is the readable view of point accuracy. A separate log-variance + exp parameterization was also tried and
was far worse (QLIKE ~126, R^2 strongly negative) because zero-inflated Parkinson variance breaks the
log-normal assumption and the exp retransform (with a large Jensen 0.5*sigma^2) amplifies errors; that
failure is specific to the log-variance/Jensen setup, not to exp on a well-scaled ratio (config C is
stable).

### VN30 and VN100 h5 (5 seeds; same panel/seed/init/split/mask across configs)
VN30 (10,013 obs) — HAR QLIKE 0.5962, MSE 2.153e-7, R^2 0.1442:
| Config | QLIKE mean +/- std | MSE mean | R^2 (ens) |
|---|---|---|---|
| A z-score+linear+floor | 0.6595 +/- 0.0428 | 2.158e-7 | 0.1444 |
| B ratio+linear+floor | 0.6126 +/- 0.0053 | 2.174e-7 | 0.1381 |
| C ratio+exp | 0.6096 +/- 0.0021 | 2.164e-7 | 0.1410 |
| D ratio+softplus | 0.6095 +/- 0.0031 | 2.166e-7 | 0.1405 |

VN100 (46,206 obs) — HAR QLIKE 0.5694, MSE 2.628e-7, R^2 0.1392:
| Config | QLIKE mean +/- std | MSE mean | R^2 (ens) |
|---|---|---|---|
| A z-score+linear+floor | 0.6021 +/- 0.0388 | 2.649e-7 | 0.1354 |
| B ratio+linear+floor | 0.5721 +/- 0.0037 | 2.642e-7 | 0.1369 |
| C ratio+exp | 0.5728 +/- 0.0026 | 2.655e-7 | 0.1335 |
| D ratio+softplus | 0.5763 +/- 0.0070 | 2.666e-7 | 0.1302 |

Date-clustered DM (QLIKE, 5-seed ensembles): D vs HAR p=0.29 (VN30), p=0.60 (VN100); C vs D p=0.83
(VN30), p=0.055 (VN100).

### Fair link comparison — bias-matched initialization (C vs D, 5 seeds)
The output-layer bias was set so the initial prediction starts at the mean ratio (~1): exp -> bias 0
(exp(0)=1); softplus -> bias log(e-1) ~ 0.5413 (softplus(0.5413)=1). Without this, softplus started at
softplus(0)=0.693, ~31% below the mean ratio — an initialization handicap for D.

| | VN30 C | VN30 D | VN100 C | VN100 D |
|---|---|---|---|---|
| QLIKE mean +/- std | 0.6140 +/- 0.0090 | 0.6104 +/- 0.0034 | 0.5729 +/- 0.0030 | 0.5737 +/- 0.0030 |
| DM C vs D (p) | 0.74 | | 0.42 | |

After bias-matching, the borderline VN100 C-advantage (p=0.055) disappears (p=0.42): exp and softplus are
not distinguishable on either panel once fairly initialized. The earlier signal was largely an
initialization artifact.

## Interpretation (statistical wording)
1. **Node-wise ratio normalization is the primary source of improved stability.** Across VN30 and VN100,
   replacing the additive z-score target with the ratio target reduced the QLIKE standard deviation from
   0.0428 to 0.0053 (VN30) and from 0.0388 to 0.0037 (VN100), with the same linear output and floor. This
   identifies node scaling, rather than the positive output link itself, as the main driver.
2. **Positive links add smaller improvements and remove the post-hoc floor.** No statistically significant
   QLIKE difference was detected between the exponential and softplus links at the 5% level on either
   panel after bias-matched initialization.
3. **Point-accuracy change is small and not systematic.** R^2 moved A->B by -0.63 pp (-4.36% relative) on
   VN30 but +0.15 pp (+1.11% relative) on VN100; there is no consistent MSE/R^2 penalty from ratio scaling.
4. **HAR retains lower mean QLIKE, but the difference is not significant.** D vs HAR: DM p=0.29 (VN30),
   p=0.60 (VN100). Failure to reject does not establish equivalence; HAR keeps a lower point estimate
   (+2.23% VN30, +1.21% VN100) but no significant QLIKE gap remains.

Conclusion: the previously observed instability of the deep model's QLIKE was largely attributable to its
target parameterization (additive standardization plus a post-hoc floor), not to its forecasting
architecture. A node-scaled ratio parameterization with a positive output link removes the instability and
the ad-hoc floor without a systematic point-accuracy cost.

## Recommended configuration
- **Primary:** C (ratio + exp, no floor) — a familiar log/exp positive link.
- **Robustness:** D (ratio + softplus, no floor) — empirically indistinguishable from C after bias-matching.
- **Ablation:** B (ratio + linear + floor) — demonstrates ratio normalization as the main component.
- **Baseline:** A (original z-score + linear + floor).

## Caveats
- QLIKE clamps the target `y==0` days to 1e-8; softplus/exp only guarantee a positive forecast, not a
  positive target.
- Single horizon (h5) shown; other horizons and a full 5-seed suite re-run under the chosen config are the
  next step before locking it into the pipeline.
- Equivalence (rather than "no significant difference") would require an equivalence test with a
  pre-specified margin; not claimed here.

## Artifacts
- Scripts: `scripts/garch_masked/{ablation_output_param.py, ablation_vn_5seed.py, exp_logvar_test.py,
  exp_softplus_test.py}` (+ tests where applicable).
- Results: `results/ablation_vn_5seed/{vn30,vn100}_h5.json` and `..._h5_bm.json`.
- Logs: `results/masked_rich_floor1e2/logs/ablation_*.log`.
