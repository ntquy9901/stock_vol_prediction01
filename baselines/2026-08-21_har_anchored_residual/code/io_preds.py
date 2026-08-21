"""Row-aligned prediction export + feature-availability manifest (plan sections 21, 26).

Every fold/model/seed writes one row-aligned table: (ticker, target_date, horizon, y_true, one column
per model prediction, alpha/gate where relevant, fold, seed). CSV is used (portable, no pyarrow needed);
``export_predictions`` accepts a dict of equal-length columns.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


def export_predictions(path: str | Path, columns: dict[str, np.ndarray]) -> None:
    """Write a dict of equal-length columns to CSV. Raises ValueError on ragged input."""
    lengths = {k: len(np.asarray(v)) for k, v in columns.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"export_predictions: unequal column lengths {lengths}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: np.asarray(v) for k, v in columns.items()}).to_csv(path, index=False)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Read back an exported prediction table."""
    return pd.read_csv(path)


def write_manifest(path: str | Path, entries: list[dict]) -> None:
    """Write the feature-availability manifest (plan section 19)."""
    cols = ["feature", "event_time", "available_time", "lookback", "fitted_on", "used_for"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for e in entries:
            w.writerow({c: e.get(c, "") for c in cols})
