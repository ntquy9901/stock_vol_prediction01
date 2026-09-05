"""Unit tests for the opt-in MLflow helper (fake mlflow injected -- no server/dependency needed)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mlflow_helper as mh  # noqa: E402


class _FakeMlflow:
    def __init__(self):
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append(("uri", uri))

    def set_experiment(self, e):
        self.calls.append(("exp", e))

    def start_run(self):
        calls = self.calls

        class _Ctx:
            def __enter__(s):
                calls.append(("start", None)); return s

            def __exit__(s, *a):
                calls.append(("end", None)); return False

        return _Ctx()

    def log_params(self, p):
        self.calls.append(("params", p))

    def log_metrics(self, m):
        self.calls.append(("metrics", m))

    def log_artifact(self, a):
        self.calls.append(("artifact", a))


def test_log_run_records_and_skips_missing_artifacts(tmp_path):
    fake = _FakeMlflow()
    art = tmp_path / "r.json"; art.write_text("{}")
    missing = tmp_path / "nope.json"
    mh.log_run("exp", {"a": 1}, {"q": 0.5}, artifacts=[str(art), str(missing)], mlflow=fake)
    assert [c[0] for c in fake.calls] == ["uri", "exp", "start", "params", "metrics", "artifact", "end"]
    assert ("artifact", str(art)) in fake.calls
    assert ("artifact", str(missing)) not in fake.calls
    assert ("params", {"a": 1}) in fake.calls and ("metrics", {"q": 0.5}) in fake.calls


def test_log_run_no_artifacts(tmp_path):
    fake = _FakeMlflow()
    mh.log_run("e", {}, {}, mlflow=fake)
    assert [c[0] for c in fake.calls] == ["uri", "exp", "start", "params", "metrics", "end"]
