# Paper layout & content rules (extracted from soict_harlstmgat_hnx.pdf)

Reusable conventions for this project's SOICT/Springer-llncs volatility papers. Extracted 2026-09-04
from the HNX version. Apply to every new paper version.

## 1. Ordering — strongest result leads ("cái tốt đưa lên trên")
- Choose the **primary/headline market = the one with the strongest positive result for the proposed
  model**, and put it first (title framing, abstract headline, Table 1, and the first Results section).
  In the HNX version, HNX led because VolGA significantly beat HAR there. Re-pick the lead market per
  paper by result strength — do NOT default to a fixed market.
- Everything after the headline market is framed as an **ablation / cross-market comparison**, ordered
  from most-to-least supportive, honestly.

## 2. Table progression — start minimal, add one variable per table
- **Table 1 = Main result, primary market, MINIMAL model set.** Only the headline comparison:
  the extended linear baseline **HAR-X**, the classical benchmark **GARCH** (if available), and the
  proposed **VolGA**. **Do NOT put LSTM in Table 1.** Columns MSE / RMSE / MAE / QLIKE, grouped by
  horizon h∈{1,5,10,22}, one row per model per horizon. (Per user: the first table compares HAR-X vs
  VolGA; the classical benchmark row is optional context.)
- **Table 2 = Graph ablation** on the primary market: **add LSTM** (HAR-X, LSTM, VolGA) so the
  VolGA−LSTM contrast isolates the graph branch's marginal value.
- **Table 3+ = Market ablations** (each other market: full set HAR-X, GARCH, LSTM, VolGA), then
  **estimator ablations** (Rogers–Satchell, windowed Yang–Zhang), then any further studies.
- Each later table introduces exactly ONE new axis (a model, a market, a target). Never dump all
  models × all markets in one table.

## 3. Table format
- Columns: `h | Model | MSE | RMSE | MAE | QLIKE`. Group rows by horizon (blank line / rule between
  horizon blocks).
- **Best value per column per horizon in bold.** Lower is better for MSE/RMSE/MAE/QLIKE.
- Learned models: QLIKE printed as `value ± per-seed std` (report the mean of seed-level metrics, not
  the seed-averaged prediction's metric).
- **Scale in the caption, not e-notation in cells:** e.g. "MSE ×10⁻⁷, RMSE/MAE ×10⁻⁴". No `1e-7` in
  prose or cells.
- Caption states: what the table is, the primary-market obs count + number of test dates at h1, and
  "Lower is better on every metric."

## 4. Section order
1. **Abstract** — two paragraphs: (1) accessible motivation (what volatility is, why forecasting it
   matters to a risk manager / option trader); (2) the model + primary market + the cross-market/
   estimator ablations + the shared 5 inputs + horizons {1,5,10,22} + benchmarks (HAR/GARCH) +
   metrics (MSE/RMSE/MAE/QLIKE) + the date-clustered Diebold–Mariano test + the one-sentence headline
   finding on the primary market.
2. **Keywords.**
3. **1 Introduction** — HAR is the dominant, hard-to-beat benchmark with a structural blind spot
   (per-stock, ignores co-movement); LSTM + GAT are the two families proposed to close it; state the
   research question ("this paper asks WHERE deep/graph models improve on HAR, across markets");
   headline finding; contributions as an explicit (i)/(ii)/(iii) list; the panel-correct
   date-clustered DM protocol (naive per-stock test inflates significance ~√(#stocks)).
4. **Terminology (for readers outside finance)** — a short glossary: Volatility, Parkinson estimator,
   HAR, GARCH, QLIKE, Diebold–Mariano, GAT, Horizon h.
5. **2 Related Work** — bold run-in lead-ins (HAR/classical, Deep & graph, Graph construction,
   Evaluation).
6. **3 Method** — Target & features (5 node features, Parkinson variance σ² at t+h); Volatility
   estimators with the published formulas (Parkinson, Rogers–Satchell, windowed Yang–Zhang);
   Models (HAR-X pooled OLS, LSTM, VolGA=LSTM+GAT, GARCH); Masked panel (keep all dates, mask unlisted
   stocks); Protocol (5 seeds, batch, epochs, floor, seed-level metric reporting, DM with HLN
   correction, the two targeted contrasts LSTM-vs-HAR-X and VolGA-vs-LSTM).
7. **Fig 1** — VolGA architecture (parallel temporal LSTM branch + spatial GAT branch → concat → MLP
   head; removing GAT gives the same-feature LSTM).
8. **4 Data** — primary market (screened ticker count) + the cross-market panels.
9. **Table 1 (main) + 5 Experiments** — state the main comparison and list the ablations.
10. **6 Results: <primary market>** — prose walking Table 1: where the proposed model wins, the DM
    p-values, where HAR-X keeps the edge. Honest.
11. **7 Ablation studies** — 7.1 Graph component (Table 2), 7.2 Other markets (Tables 3+),
    7.3 Alternative estimators.
12. **8 Discussion** — inference-design note (why date-clustered DM), capacity note (VolGA adds
    params), the where/when-it-helps synthesis.
13. **9 Limitations** — not split-adjusted prices; variance (σ²) target; QLIKE floor sensitivity;
    small-N markets; single-country scope.
14. **10 Conclusion.**

## 5. Content/voice conventions (objective, reviewer-facing)
- Honest "where/when it helps" framing — never "hurts"/"never helps"/"fails". State effects with the
  DM evidence; a non-significant difference is reported as such, not spun.
- HAR is stated as a deliberately-simple, famously-hard-to-beat benchmark.
- Report ALL horizons {1,5,10,22}. No directional-accuracy (DirAcc) claims.
- Primary target is the Parkinson **variance** (σ², a squared quantity), stated explicitly.
- Data-quality limitations disclosed openly (zero-range days, unverified split adjustment) rather than
  presenting the data as uniformly clean.
- Named estimators/metrics use their published formulas (Parkinson, Rogers–Satchell, Yang–Zhang, QLIKE,
  Diebold–Mariano) with citations; HAR-X = HAR + exogenous regressors (Corsi 2009; Clements et al. 2024).
- Every reported number comes from a stored results JSON; state this.
