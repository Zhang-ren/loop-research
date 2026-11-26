import json
import subprocess
import sys
import venv
from pathlib import Path

import pytest
from typer.testing import CliRunner

from loop_research import cli as cli_module
from loop_research.cli import app, read_project
from loop_research.engine import Research
from loop_research.study_intake import Finding, IntakeFindings, prepare


def commit(repo):
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Fixture input"], check=True)


@pytest.fixture
def existing_experiment(tmp_path):
    repo = tmp_path / "existing-experiment"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "splits.json").write_text('{"train":["T"],"validation":["V"],"test":["H"]}')
    (repo / "evaluate.py").write_text('from pathlib import Path\nimport os\n'
        'Path(os.environ["LOOP_METRICS"]).write_text(\'{"metrics":{"quality":0.5}}\')\n')
    (repo / "README.md").write_text("The split is tracked in splits.json for this example.")
    commit(repo)
    return repo


def test_preparation_persists_questions_checks_dependencies_and_only_then_measures(
        tmp_path, existing_experiment, monkeypatch):
    monkeypatch.setattr("loop_research.study_intake.discover", lambda *_: ([], []))
    repo, root = existing_experiment, tmp_path / "run"
    spec = tmp_path / "input.json"
    spec.write_text(json.dumps({"objective": "Improve fixture classification", "command_experiment": {
        "repository": str(repo), "command": [sys.executable, "evaluate.py"], "split_manifest": "splits.json",
        "baseline": {"threshold": 0}, "choices": {"threshold": [0, 1]}}}))
    cli = CliRunner()
    response = cli.invoke(app, ["prepare-study", str(repo), str(root), "--spec", str(spec), "--non-interactive"])
    assert response.exit_code == 3, response.output
    issues = json.loads(response.output)["issues"]
    assert {"metrics", "max_experiments", "knowledge"} <= {item["field"] for item in issues}
    status = cli.invoke(app, ["status", str(root)])
    assert json.loads(status.output)["status"] == "awaiting_input"
    assert not (root / "state.json").exists() and not (root / "baseline-logs").exists()

    answers = tmp_path / "answers.json"
    asset = tmp_path / "required-input.txt"
    answers.write_text(json.dumps({"metrics": [{"name": "quality", "threshold": 0.9}],
        "knowledge": "Synthetic inputs and a known measurement protocol.", "max_experiments": 2,
        "affordable": True, "reversible": True, "confirmed_knowledge": True,
        "environment.local_paths": [str(asset)], "environment.python_modules": ["json"]}))
    args = ["prepare-study", str(repo), str(root), "--non-interactive"]
    response = cli.invoke(app, [*args, "--answers", str(answers)])
    assert response.exit_code == 3 and "environment.local_paths" in response.output
    assert not (root / "study-spec.json").exists()
    asset.write_text("fixture")
    response = cli.invoke(app, args)
    assert response.exit_code == 0, response.output
    ready = read_project(root / "study-spec.json")
    assert len(ready.command_experiment.revision) == 40
    assert not (root / "baseline-logs").exists()
    asset.rename(asset.with_suffix(".moved"))
    with pytest.raises(ValueError, match="实验环境已变"):
        Research.initialize(root, ready)
    assert not (root / "baseline-logs").exists()

    asset.write_text("fixture restored")
    assert cli.invoke(app, args).exit_code == 0
    response = cli.invoke(app, ["init", str(root / "study-spec.json"), str(root)])
    assert response.exit_code == 0, response.output
    assert Research(root).store.read().best_metrics == {"quality": 0.5}


def test_discovery_accepts_cited_fields_and_does_not_accept_invented_source(tmp_path, existing_experiment):
    class Worker:
        def run(self, role, payload, output):
            assert role == "study-intake"
            return IntakeFindings(proposed=[
                Finding(field="command_experiment.split_manifest", value_json='"splits.json"', path="README.md",
                        quote="tracked in splits.json"),
                Finding(field="metrics", value_json='["quality"]', path="README.md", quote="tracked in splits.json"),
                Finding(field="metrics", value_json='[{"name":"quality","threshold":1}]', path="README.md",
                        quote="achieve 100 percent")], notes=["Missing scoring threshold."]), {}

    run = tmp_path / "run"
    (existing_experiment / "experiment.json").write_text(json.dumps({"objective": "Run the local fixture"}))
    result = prepare(existing_experiment, run, worker=Worker())
    assert result["status"] == "awaiting_input"
    assert {item["field"] for item in result["issues"]} >= {"metrics", "command_experiment.command"}
    assert result["draft"]["command_experiment"]["split_manifest"] == "splits.json"
    assert [source["field"] for source in result["sources"]] == ["command_experiment.split_manifest"]


def test_preflight_blocks_installed_dependency_that_fails_to_import(tmp_path, existing_experiment):
    from loop_research.preflight import inspect

    environment = tmp_path / "experiment-python"
    venv.EnvBuilder(with_pip=False).create(environment)
    runner = environment / "bin" / "python"
    site_packages = subprocess.check_output([str(runner), "-I", "-c",
        "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
    dependency = Path(site_packages) / "experiment_dependency.py"
    dependency.write_text("raise ImportError('fixture for an unavailable native library')\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"name": "Fixture", "objective": "Verify the existing fixture evaluator",
        "knowledge": "Fixture metrics and data split are fixed.", "archive_uri": "local",
        "metrics": [{"name": "quality", "threshold": 0.9}], "max_experiments": 3,
        "reversible": True, "affordable": True, "confirmed_knowledge": True,
        "environment": {"python_modules": ["experiment_dependency"]},
        "command_experiment": {"repository": str(existing_experiment), "command": [str(runner), "evaluate.py"],
            "split_manifest": "splits.json", "baseline": {"a": 0, "b": 0, "c": 0},
            "choices": {"a": [0, 1], "b": [0, 1], "c": [0, 1]}}}))
    project = read_project(spec)
    blocked = inspect(project, check_codex=False)
    assert blocked["status"] == "blocked"
    assert [(item["field"], item["detail"]) for item in blocked["issues"]] == [
        ("environment.python_modules", "experiment_dependency")]

    dependency.write_text("ready = True\n")
    assert inspect(project, check_codex=False)["status"] == "ready"


def test_start_resumes_after_missing_input_and_real_baseline_failure(tmp_path, existing_experiment, monkeypatch):
    repo, root = existing_experiment, tmp_path / "failed"
    spec = tmp_path / "spec.json"
    asset = tmp_path / "required_input"
    spec.write_text(json.dumps({"objective": "Verify the existing evaluator", "knowledge": "Fixture metrics and split are fixed.",
        "metrics": [{"name": "wrong_metric", "threshold": 0.9}], "max_experiments": 1,
        "reversible": True, "affordable": True, "confirmed_knowledge": True,
        "environment": {"local_paths": [str(asset)]},
        "command_experiment": {"command": [sys.executable, "missing.py"],
                               "split_manifest": "splits.json", "baseline": {"a": 0}, "choices": {"a": [0, 1]}}}))
    monkeypatch.setattr("loop_research.study_intake.discover", lambda *_: ([], []))
    dispatches = []

    def check_dispatch(run_root, **_):
        dispatches.append(Research(run_root).store.read().best_metrics)

    monkeypatch.setattr(cli_module, "run", check_dispatch)
    cli = CliRunner()
    args = ["prepare-study", str(repo), str(root), "--non-interactive", "--start"]
    blocked = cli.invoke(app, [*args, "--spec", str(spec)])
    assert blocked.exit_code == 3, blocked.output
    assert {i["field"] for i in json.loads(blocked.output)["issues"]} == {
        "command_experiment.command", "environment.local_paths"}
    assert not (root / "baseline-logs").exists() and not dispatches

    asset.write_text("restored")
    answers = tmp_path / "answers.json"
    answers.write_text(json.dumps({"command_experiment.command": [sys.executable, "evaluate.py"]}))
    revised = cli.invoke(app, [*args, "--answers", str(answers)])
    assert revised.exit_code == 3
    assert {i["field"] for i in json.loads(revised.output)["issues"]} == {"reversible", "confirmed_knowledge"}
    answers.write_text(json.dumps({"reversible": True, "confirmed_knowledge": True}))
    failed = cli.invoke(app, [*args, "--answers", str(answers)])
    assert failed.exit_code == 3, failed.output
    status = json.loads(failed.output)
    assert status == json.loads(cli.invoke(app, ["status", str(root)]).output)
    assert status["status"] == "awaiting_input"
    assert any(i["field"] == "metrics" and i["detail"] == "wrong_metric" for i in status["issues"])
    assert (root / "baseline-logs" / "metrics.json").exists() and not dispatches
    assert not (root / "state.json").exists() and not list((root / "jobs").glob("*/request.json"))

    answers.write_text(json.dumps({"metrics": [{"name": "quality", "threshold": 0.9}],
                                   "confirmed_knowledge": True}))
    completed = cli.invoke(app, [*args, "--answers", str(answers)])
    assert completed.exit_code == 0, completed.output
    assert '"status": "baseline_passed"' in completed.stdout
    assert dispatches == [{"quality": 0.5}]
