"""Tests for the AUD-015 provenance helper in src/common/provenance.py.

Background: results.json files written by training scripts don't record
which git commit / repo state produced a given number. get_provenance()
adds a minimal, never-raising dict (git_sha, git_dirty, timestamp) meant
to be stored under results["provenance"].
"""
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.common.provenance import get_provenance

pytestmark = pytest.mark.smoke


class TestGetProvenanceRealRepo:
    def test_returns_expected_keys(self):
        result = get_provenance()
        assert set(result.keys()) == {"git_sha", "git_dirty", "timestamp"}

    def test_git_sha_is_real_40_char_sha_in_this_repo(self):
        # This repo IS a git checkout, so we should get a real SHA, not "unknown".
        result = get_provenance()
        assert result["git_sha"] != "unknown"
        assert len(result["git_sha"]) == 40
        assert all(c in "0123456789abcdef" for c in result["git_sha"])

    def test_git_dirty_is_bool(self):
        result = get_provenance()
        assert isinstance(result["git_dirty"], bool)

    def test_timestamp_is_iso_format_utc(self):
        from datetime import datetime
        result = get_provenance()
        parsed = datetime.fromisoformat(result["timestamp"])
        assert parsed.tzinfo is not None


class TestGetProvenanceFallback:
    def test_falls_back_to_unknown_when_git_not_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            result = get_provenance()
        assert result["git_sha"] == "unknown"
        assert result["git_dirty"] is False

    def test_falls_back_when_not_a_git_checkout(self):
        # Simulate `git rev-parse HEAD` failing (non-zero exit -> CalledProcessError)
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),
        ):
            result = get_provenance()
        assert result["git_sha"] == "unknown"
        assert result["git_dirty"] is False

    def test_falls_back_on_timeout(self):
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=10)):
            result = get_provenance()
        assert result["git_sha"] == "unknown"
        assert result["git_dirty"] is False

    def test_never_raises_even_on_unexpected_error(self):
        with patch("subprocess.run", side_effect=RuntimeError("unexpected")):
            result = get_provenance()  # must not raise
        assert result["git_sha"] == "unknown"
        assert result["git_dirty"] is False
