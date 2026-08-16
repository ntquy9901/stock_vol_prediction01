"""Minimal run-provenance capture for results.json files (AUD-015).

Background: training scripts write results.json without recording which
code version / repo state produced the numbers, making it hard to later
trace a cited number back to an exact commit. This module adds a single
small helper, `get_provenance()`, meant to be stored under a
`results["provenance"]` key by callers.

Kept deliberately minimal per CLAUDE.md Simplicity First: no config-hash,
no schema versioning — just git SHA + dirty flag + timestamp. A training
run must never fail because of this feature, so git lookups are wrapped
in try/except and fall back to "unknown".
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# Repo root = parent of src/common/ (this file lives at src/common/provenance.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(args: list) -> str:
    """Run a git command with cwd=repo root and return stripped stdout.

    Raises on any failure (non-git checkout, git not installed, non-zero
    exit) so the caller can decide the fallback value.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return result.stdout.strip()


def get_provenance() -> Dict[str, object]:
    """Return a small dict of run-provenance info for embedding in results.json.

    Returns:
        {"git_sha": <40-char sha or "unknown">,
         "git_dirty": <bool>,
         "timestamp": <ISO-8601 UTC string>}

    Never raises: any failure in git lookup (e.g. run outside a git
    checkout, git not on PATH) falls back to "unknown" / False rather
    than crashing a training run.
    """
    try:
        git_sha = _run_git(["rev-parse", "HEAD"])
    except Exception:
        git_sha = "unknown"

    try:
        git_dirty = bool(_run_git(["status", "--porcelain"]))
    except Exception:
        git_dirty = False

    return {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
