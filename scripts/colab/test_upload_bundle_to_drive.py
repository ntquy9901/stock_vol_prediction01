"""Unit tests for the rclone Drive-upload helper."""
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import upload_bundle_to_drive as ud  # noqa: E402


def test_rclone_copy_cmd():
    cmd = ud.rclone_copy_cmd(Path("/x/bundle.zip"), "gdrive:luanvan/data/")
    assert cmd[:2] == ["rclone", "copy"]
    assert "gdrive:luanvan/data/" in cmd
    assert str(Path("/x/bundle.zip")) in cmd


def test_upload_runs_rclone_and_returns_rc(tmp_path):
    bundle = tmp_path / "b.zip"; bundle.write_bytes(b"x")
    seen = {}

    def fake_run(cmd):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    rc = ud.upload(bundle, "gdrive:luanvan/data/", run=fake_run)
    assert rc == 0
    assert seen["cmd"] == ud.rclone_copy_cmd(bundle, "gdrive:luanvan/data/")


def test_upload_propagates_nonzero_rc(tmp_path):
    bundle = tmp_path / "b.zip"; bundle.write_bytes(b"x")
    rc = ud.upload(bundle, "gdrive:x/", run=lambda cmd: types.SimpleNamespace(returncode=3))
    assert rc == 3


def test_upload_missing_bundle_raises(tmp_path):
    with pytest.raises(SystemExit):
        ud.upload(tmp_path / "nope.zip")
