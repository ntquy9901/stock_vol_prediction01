"""Shared fixtures for verify_audit_fixes tests.

Every gate test that needs git or pytest runs them for real (subprocess)
against a tiny fixture repository under tmp_path — never mocked — because
these are I/O runners, not pure helpers (CLAUDE.md: "test the I/O runner,
not just pure helpers").
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A minimal, clean, committed git repository."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", "-b", "master", cwd=repo)
    _git("config", "user.email", "test@example.com", cwd=repo)
    _git("config", "user.name", "Test", cwd=repo)
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git("add", "README.md", cwd=repo)
    _git("commit", "-q", "-m", "initial commit", cwd=repo)
    return repo


@pytest.fixture
def evidence_dir(tmp_path: Path) -> Path:
    d = tmp_path / "evidence" / "2026-01-01_000000"
    d.mkdir(parents=True)
    return d
