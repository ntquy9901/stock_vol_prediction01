"""Opt-in MLflow experiment-tracking helper (per user 2026-09-05).

Does NOT touch existing training code — call ``log_run(...)`` from a NEW run when you want a queryable
record of params/metrics/artifacts. Defaults to a local file backend (``mlruns/`` in the repo), so it
needs no server. Wire it into a driver when wanted, e.g.::

    from scripts.tracking.mlflow_helper import log_run
    log_run("edge_hmatched", params={"market": "sp500_clean", "horizon": 1, "seeds": 5},
            metrics={"qlike_VolGA_hm": 0.51}, artifacts=["results/edge_hmatched/edgehm_sp500_clean_h1.json"])

Then browse locally with ``mlflow ui`` (reads ``mlruns/``).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_TRACKING_URI = "file:./mlruns"


def _import_mlflow():  # pragma: no cover - thin import shim (real dependency)
    import mlflow
    return mlflow


def log_run(experiment: str, params: dict, metrics: dict, artifacts=None,
            tracking_uri: str = DEFAULT_TRACKING_URI, mlflow=None) -> None:
    """Log one MLflow run: set the experiment, then record params, metrics and any artifact files.

    ``mlflow`` may be injected (for tests); when None the real ``mlflow`` module is imported lazily so
    importing this helper never forces the heavy mlflow dependency. Missing artifact paths are skipped
    (logged runs should not fail because an optional dump is absent)."""
    m = mlflow if mlflow is not None else _import_mlflow()
    m.set_tracking_uri(tracking_uri)
    m.set_experiment(experiment)
    with m.start_run():
        m.log_params(params)
        m.log_metrics(metrics)
        for a in (artifacts or []):
            if Path(a).exists():
                m.log_artifact(a)
