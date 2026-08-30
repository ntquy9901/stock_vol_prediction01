# Walk-forward HAR-X vs no-graph LSTM — all horizons (VN100)

Extends the h1 walk-forward (expanding-window, retrain every K=66, val=66, 5 seeds, 16ep) to h5/h10/h22
via the same gated runner. Pooled-OOS QLIKE + date-clustered DM (LSTM vs HAR-X). ~7 folds each, 102 nodes.

| h | HAR-X QLIKE | LSTM QLIKE | Δ (LSTM−HARX) | walk-forward DM p | fixed-split DM p (real) |
|---:|---:|---:|---:|---|---|
| 1 | 0.5074 | **0.4965** | −0.0109 | 0.372 (n.s.) | **1.1e-3 — HAR-X wins** |
| 5 | 0.5618 | 0.5644 | +0.0026 | 0.826 (n.s.) | 0.40 (n.s.) |
| 10 | 0.6006 | 0.6001 | −0.0005 | 0.963 (n.s.) | 0.73 (n.s.) |
| 22 | 0.6386 | 0.6474 | +0.0088 | 0.658 (n.s.) | 0.41 (n.s.) |

## Conclusion
Under periodic retraining the LSTM is **statistically equivalent to HAR-X at every horizon** (all DM
p>0.05). The only place the fixed split gave HAR-X a *significant* edge was h1 (p=1.1e-3); walk-forward
removes exactly that deficit (p=0.372) — so the LSTM's fixed-split h1 inferiority was a single-training
artifact. h5/h10/h22 were already ties under both protocols. Point estimates are a wash (LSTM edges h1/h10,
HAR-X edges h5/h22). **Net: retraining brings the deep model to full equivalence with HAR-X across horizons
— a genuine fix, but not superiority; beating HAR-X would need another lever (QLIKE-loss training / richer
inputs).**

## Caveats
- The `fixed_split_reference` field inside the h5/h10/h22 result.json is hard-coded to the h1 reference
  (p=1.1e-3) — a known runner limitation; the REAL fixed-split DM for h5/h10/h22 (from the delivered
  masked_rich_floor1e2) is p=0.40/0.73/0.41 (all n.s.), used in the table above.
- DM is date-clustered but not fully HAC-corrected across fold boundaries — p-values approximate.
- Single dataset (VN100); h1 learning curves confirm convergence (early-stop 34/35, median best-epoch 6).
