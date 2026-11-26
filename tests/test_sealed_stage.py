import subprocess
import sys
from pathlib import Path

import pytest

from loop_research.sealed_stage import file_hash, run_plan
from loop_research.storage import canonical


def make_plan(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    scratch = tmp_path / "replica"
    scratch.mkdir()
    script = scratch / "step.py"
    script.write_text("# A sealed stage")
    value = {"project": str(project), "scratch": str(scratch), "interpreter": sys.executable,
             "files": {script.name: file_hash(script)}, "batches": [
                 [{"name": name, "script": script.name, "arguments": [name], "outputs": [name + ".json"]}]
                 for name in ("prepare", "check")
             ]}
    path = tmp_path / "plan.json"
    path.write_bytes(canonical(value))
    return path, value


def test_reopen_and_reject_changed_output(tmp_path, monkeypatch):
    path, value = make_plan(tmp_path)
    launches = []

    def execute(interpreter, scratch, project, script, arguments, **kwargs):
        return [interpreter, str(scratch), *arguments]

    def run(command, **kwargs):
        launches.append(command[-1])
        (Path(command[1]) / (command[-1] + ".json")).write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("loop_research.sealed_stage.stage_command", execute)
    monkeypatch.setattr("loop_research.sealed_stage.subprocess.run", run)
    work = tmp_path / "receipts"
    assert run_plan(path, work)["status"] == "passed"
    assert run_plan(path, work)["status"] == "passed"
    assert launches == ["prepare", "check"]
    (Path(value["scratch"]) / "prepare.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="completed stage output has changed"):
        run_plan(path, work)


def test_failed_stage_prevents_later_executions(tmp_path, monkeypatch):
    path, _ = make_plan(tmp_path)
    launches = []
    monkeypatch.setattr("loop_research.sealed_stage.stage_command", lambda *args, **kwargs: args[4])

    def fail(command, **kwargs):
        launches.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr("loop_research.sealed_stage.subprocess.run", fail)
    work = tmp_path / "receipts"
    for _ in range(2):
        record = run_plan(path, work)
        assert record["status"] == "incomplete" and record["stages"] == {"prepare": "failed"}
    assert launches == [["prepare"]]
