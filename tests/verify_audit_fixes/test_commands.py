import sys
from pathlib import Path

from scripts.verify_audit_fixes.commands import (
    append_command_output,
    run_command,
    write_command_output,
)


def test_run_command_success_captures_stdout_and_exit_code(tmp_path):
    result = run_command([sys.executable, "-c", "print('hello')"], cwd=tmp_path)
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.duration_seconds >= 0


def test_run_command_nonzero_exit_is_captured_not_raised(tmp_path):
    result = run_command([sys.executable, "-c", "import sys; sys.exit(3)"], cwd=tmp_path)
    assert result.exit_code == 3


def test_run_command_stderr_captured(tmp_path):
    result = run_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(1)"], cwd=tmp_path
    )
    assert result.exit_code == 1
    assert "boom" in result.stderr


def test_run_command_missing_executable_is_captured_not_raised(tmp_path):
    result = run_command(["this-executable-does-not-exist-xyz"], cwd=tmp_path)
    assert result.exit_code == -2
    assert "COMMAND NOT FOUND" in result.stderr


def test_run_command_timeout_is_captured_not_raised(tmp_path):
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=1
    )
    assert result.exit_code == -1
    assert "TIMEOUT" in result.stderr


def test_write_command_output_creates_file_with_all_fields(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    result = run_command([sys.executable, "-c", "print('x')"], cwd=tmp_path)
    path = write_command_output(evidence_dir, "out.txt", result)
    text = path.read_text(encoding="utf-8")
    assert "exit_code: 0" in text
    assert "x" in text
    assert result.command in text


def test_append_command_output_appends_after_write(tmp_path):
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    r1 = run_command([sys.executable, "-c", "print('first')"], cwd=tmp_path)
    r2 = run_command([sys.executable, "-c", "print('second')"], cwd=tmp_path)
    write_command_output(evidence_dir, "out.txt", r1)
    append_command_output(evidence_dir, "out.txt", r2)
    text = (evidence_dir / "out.txt").read_text(encoding="utf-8")
    assert "first" in text
    assert "second" in text
    assert text.index("first") < text.index("second")
