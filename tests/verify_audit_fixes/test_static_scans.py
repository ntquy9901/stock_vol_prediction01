from pathlib import Path

from scripts.verify_audit_fixes.static_scans import (
    format_scan_report,
    iter_scan_files,
    run_all_scans,
    scan_bare_except,
    scan_duplicate_module_names,
    scan_hardcoded_paths,
    scan_random_split,
)


def _write(root: Path, rel: str, content: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_iter_scan_files_excludes_vendored_dirs(tmp_path):
    _write(tmp_path, "src/ok.py", "x = 1\n")
    _write(tmp_path, "archive/skip.py", "x = 1\n")
    _write(tmp_path, ".claude/skip.py", "x = 1\n")
    _write(tmp_path, "data/skip.py", "x = 1\n")
    _write(tmp_path, ".git/skip.py", "x = 1\n")

    files = iter_scan_files(tmp_path)
    rels = {str(f.relative_to(tmp_path)).replace("\\", "/") for f in files}
    assert rels == {"src/ok.py"}


def test_scan_hardcoded_paths_flags_windows_and_unix_absolute_paths(tmp_path):
    f = _write(
        tmp_path,
        "src/bad.py",
        'a = "C:\\\\Users\\\\me\\\\file.txt"\n'
        'b = "/home/me/data"\n'
        'c = "relative/path.txt"\n',
    )
    matches = scan_hardcoded_paths([f], tmp_path)
    lines = {m["line"] for m in matches}
    assert 1 in lines
    assert 2 in lines
    assert 3 not in lines


def test_scan_bare_except_flags_bare_except_only(tmp_path):
    f = _write(
        tmp_path,
        "src/bad.py",
        "try:\n"
        "    pass\n"
        "except:\n"
        "    pass\n"
        "try:\n"
        "    pass\n"
        "except ValueError:\n"
        "    pass\n",
    )
    matches = scan_bare_except([f], tmp_path)
    assert [m["line"] for m in matches] == [3]


def test_scan_random_split_flags_random_split_and_train_test_split(tmp_path):
    f = _write(
        tmp_path,
        "src/bad.py",
        "from torch.utils.data import random_split\n"
        "train, test = random_split(dataset, [0.8, 0.2])\n"
        "from sklearn.model_selection import train_test_split\n"
        "a, b = train_test_split(x, y)\n"
        "splitter = TemporalSplitter()\n",
    )
    matches = scan_random_split([f], tmp_path)
    lines = {m["line"] for m in matches}
    assert lines == {2, 4}


def test_scan_duplicate_module_names_flags_same_basename_in_different_dirs(tmp_path):
    f1 = _write(tmp_path, "baselines/a/code/train.py", "x = 1\n")
    f2 = _write(tmp_path, "baselines/b/code/train.py", "x = 1\n")
    _write(tmp_path, "baselines/a/code/__init__.py", "")
    _write(tmp_path, "baselines/b/code/__init__.py", "")

    dups = scan_duplicate_module_names([f1, f2, tmp_path / "baselines/a/code/__init__.py"], tmp_path)
    assert len(dups) == 1
    assert dups[0]["module_name"] == "train.py"
    assert dups[0]["count"] == 2


def test_scan_duplicate_module_names_no_duplicates_for_unique_names(tmp_path):
    f1 = _write(tmp_path, "src/a.py", "x = 1\n")
    f2 = _write(tmp_path, "src/b.py", "x = 1\n")
    dups = scan_duplicate_module_names([f1, f2], tmp_path)
    assert dups == []


def test_run_all_scans_returns_all_keys(tmp_path):
    _write(tmp_path, "src/clean.py", "x = 1\n")
    result = run_all_scans(tmp_path)
    assert set(result.keys()) == {
        "files_scanned",
        "hardcoded_paths",
        "bare_except",
        "random_split",
        "duplicate_module_names",
    }
    assert result["files_scanned"] == 1


def test_format_scan_report_is_readable_text(tmp_path):
    _write(tmp_path, "src/bad.py", 'a = "C:\\\\Users\\\\x"\n')
    result = run_all_scans(tmp_path)
    report = format_scan_report(result)
    assert "hardcoded_paths" in report
    assert "src/bad.py:1" in report
