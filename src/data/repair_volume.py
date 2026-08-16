"""Repair spurious zero-volume days in raw ``*_ohlcv.csv`` (any universe: VN30/VN100/HOSE/HNX).

Some feeds (e.g. Yahoo for less-liquid mid-caps) report ``volume == 0`` on a day where the price
actually MOVED (``high != low``) — an inconsistency, since a non-zero intraday range implies trades.
This repairs ONLY those glitch rows by linearly interpolating the volume from the nearest valid
trading days (leading/trailing glitches are filled from the available side). Legitimate no-trade days
(``volume == 0`` AND ``high == low``, e.g. limit-locked or halted sessions) are LEFT AS 0 — those
zeros are real. Only the ``volume`` column is touched; OHLC and the Parkinson target are unaffected.

Idempotent: after repair there are no ``volume == 0 & high != low`` rows, so re-running changes
nothing.

CLI:
  python -m src.data.repair_volume data/raw/prices/vn100      # VN100
  python -m src.data.repair_volume data/raw/prices            # VN30
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def repair_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return ``(repaired_df, n_rows_repaired)``. Repairs volume==0 rows where high != low only."""

    out = df.copy()
    vol = out["volume"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    glitch = (vol == 0) & (high != low)      # price moved but zero volume -> spurious
    if not glitch.any():
        return out, 0

    repaired = (
        vol.mask(glitch)                      # blank the glitch volumes
        .interpolate(method="linear", limit_direction="both")   # fill from neighbouring trading days
        .fillna(0.0)                          # all-glitch edge case -> leave 0
        .round()
        .astype("int64")
    )
    n = int((repaired.to_numpy() != out["volume"].to_numpy()).sum())
    out["volume"] = repaired
    return out, n


def repair_file(path: Path) -> int:
    df = pd.read_csv(path, dtype={"date": str})   # keep the date column verbatim
    repaired, n = repair_frame(df)
    if n:
        repaired.to_csv(path, index=False)
    return n


def repair_dir(directory: Path) -> dict[str, int]:
    return {p.stem.replace("_ohlcv", ""): repair_file(p)
            for p in sorted(Path(directory).glob("*_ohlcv.csv"))}


def main(directory: str) -> int:
    res = repair_dir(Path(directory))
    changed = {k: v for k, v in res.items() if v}
    for ticker, n in sorted(changed.items(), key=lambda kv: -kv[1]):
        print(f"{ticker}: {n} glitch-volume day(s) repaired")
    print(f"repaired {len(res)} files; {len(changed)} changed; {sum(res.values())} rows total")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/prices"))
