"""Single source of truth for this baseline's tunable constants (no scattered magic numbers, per the
project config-hardcode rule). Shared windows/floors come from the canonical ``pipeline_config``; only
the limit-lock-specific constants live here."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "submission" / "soict_lstm_gat"))

import pipeline_config as pc  # noqa: E402

# HOSE (VN30/VN100) daily price limit is +/-7%; use a value just under it so a bar that closes exactly at
# the limit is caught despite rounding/reference-price differences.
LIMIT_FRAC: float = 0.065  # config-ok: this baseline's canonical config module (the designated place for it)
# "near-limit" day = |daily_return| >= LIMIT_FRAC * NEAR_LIMIT_MULT (captures approach-to-limit pressure).
NEAR_LIMIT_MULT: float = 0.9  # config-ok: baseline canonical config constant
# trailing window for the rolling lock / near-limit frequencies = the project monthly convention (22).
LOCK_WINDOW: int = pc.HAR_MONTHLY_WINDOW
# override-probability sweep: on test cells with P(lock) >= threshold, pull the forecast to the floor.
LOCK_THRESHOLDS: tuple = (0.5, 0.7, 0.9)  # config-ok: baseline canonical config constant (sweep)
