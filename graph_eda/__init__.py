"""Leakage-safe EDA for graph/GNN justification on daily VN30 Parkinson volatility.

Executes the experiment plan in
``docs/eda_guide/parkinson_volatility_gnn_eda_experiment_plan.md``.

Target semantics (explicit, per plan section 3):
- ``pk_var`` = Parkinson VARIANCE (the project's target, matches
  ``data/processed/*_processed.csv`` ``parkinson_variance`` column).
- ``pk_vol`` = sqrt(pk_var) = Parkinson VOLATILITY (the plan's default ``pk_vol``).
Both are retained; every output states which is used.
"""
