"""Exercise the real isolated executor with reproducible, labeled test workers."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from test_command_project import CommandWorker
from test_workflow import RadarDouble

from loop_research.command_candidate import STRUCTURE_TARGET, candidate_code, candidate_parts
from loop_research.engine import Research
from loop_research.followup import refuted
from loop_research.models import CodeResult, IndependentReview, Project, StructuralReview
from loop_research.replay import reproduce
from loop_research.workers import WorkerFailure, codex_executable

BASE = """class Classifier:
    def __init__(self, value):
        self.value = value
    def fit(self, x, y):
        return self
    def predict(self, x):
        return [a & (1 - b) if self.value == 1 else 0 for a, b in x]
def build_model(seed, parameters):
    return Classifier(parameters["value"])
"""
STRUCTURE = BASE.replace("[a & (1 - b) if self.value == 1 else 0 for a, b in x]", "[a ^ b for a, b in x]")
PREDICT = """import json, sys, time
from pathlib import Path
from model import build_model

inputs = json.loads(Path(sys.argv[1]).read_text())
assert set(inputs) == {"train", "features", "parameters", "seed"}
assert not Path("samples.json").exists() and not Path("score.py").exists()
assert all(isinstance(row, list) for row in inputs["features"])
time.sleep(0.6)
model = build_model(inputs["seed"], inputs["parameters"])
model.fit([i["features"] for i in inputs["train"]], [i["label"] for i in inputs["train"]])
Path(sys.argv[2]).write_text(json.dumps({"predictions": model.predict(inputs["features"])}))
"""
SCORE = """import json, sys
from pathlib import Path

inputs = json.loads(Path(sys.argv[1]).read_text())
assert set(inputs) == {"labels", "predictions"}
assert not Path("model.py").exists() and not Path("predict.py").exists()
matches = sum(a == b for a, b in zip(inputs["labels"], inputs["predictions"]))
Path(sys.argv[2]).write_text(json.dumps({"metrics": {"quality": matches / len(inputs["labels"])}}))
"""


class SingleSourceRadar(RadarDouble):
    def search(self, queries, **kwargs):
        result = super().search(queries)
        result["source_ids"] = result["source_ids"][:1]
        return result


class EscapeWorker(CommandWorker):
    def run(self, role, payload, response, **kwargs):
        meta = {"TEST_DOUBLE": True, "role": role}
        if role == "coder":
            hypothesis = payload["hypothesis"]
            variant = json.loads(payload["baseline_code"])
            if hypothesis["machine_target"] == STRUCTURE_TARGET:
                variant["model"] = STRUCTURE
            else:
                variant["parameters"]["value"] = 1
            return CodeResult(code=json.dumps(variant), rationale="TEST code intervention",
                              source_ids=hypothesis["source_ids"], changed_axis=hypothesis["machine_target"]), meta
        if role == "reviewer":
            # Deliberately choose the weaker first-round candidate to exercise the disagreement route.
            return IndependentReview(selected_evidence=next(iter(payload["scores"])),
                                     evidence_ids=payload["evidence_ids"],
                                     rationale="TEST controlled disagreement or agreement", concerns=[]), meta
        if role == "structural-review":
            return StructuralReview(accepted=True, rationale="TEST fixture model checked", findings=[]), meta
        return super().run(role, payload, response, **kwargs)


@pytest.fixture
def prediction_project(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("Isolated prediction uses Linux bubblewrap")
    if not shutil.which("bwrap"):
        try:
            bundled = Path(codex_executable()).parent.parent / "codex-resources" / "bwrap"
        except (OSError, WorkerFailure):
            pytest.skip("bubblewrap unavailable")
        if not os.access(bundled, os.X_OK):
            pytest.skip("bubblewrap unavailable")
    repo = tmp_path / "source"
    repo.mkdir()
    splits = {"train": ["t0", "t1"], "validation": ["v0", "v1", "v2", "v3", "v4"], "test": ["z0", "z1"]}
    features = [(0, 0), (1, 0), (0, 0), (1, 1), (1, 0), (0, 1), (1, 0), (1, 0), (1, 1)]
    rows = {
        key: {"features": point, "label": point[0] ^ point[1]}
        for key, point in zip(sum(splits.values(), []), features, strict=True)
    }
    files = {"splits.json": json.dumps(splits), "samples.json": json.dumps(rows),
             "model.py": BASE, "predict.py": PREDICT, "score.py": SCORE}
    for name, content in files.items():
        (repo / name).write_text(content)
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Fixture data, split and scoring"], check=True)
    return Project.model_validate({
        "name": "Isolated prediction fixture", "objective": "Improve the synthetic XOR classification quality",
        "knowledge": "Labeled integration fixture for exercising the isolated prediction contract.",
        "archive_uri": "local", "metrics": [{"name": "quality", "threshold": 1}], "max_experiments": 3,
        "branches": 1, "stagnation_batches": 1, "independent_review": True, "followup_experiments": True,
        "reversible": True, "affordable": True, "confirmed_knowledge": True,
        "command_experiment": {"repository": str(repo), "command": ["isolated-predict"], "split_manifest": "splits.json",
                               "bridge_interpreter": sys.executable, "baseline": {"value": 0}, "choices": {"value": [0, 1, 2]}},
    })


def drive(engine, phase):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        state = engine.tick()
        if state.phase == phase:
            return state
        time.sleep(0.05)
    raise AssertionError(engine.store.read())


def test_isolated_escape_review_crash_recovery_and_holdout(tmp_path, prediction_project):
    root = tmp_path / "run"
    engine = Research.initialize(root, prediction_project)
    engine.worker = engine.reviewer = EscapeWorker()
    engine.radar = SingleSourceRadar(engine.evidence)
    pending = drive(engine, "review")
    assert pending.stop_reason == "independent_review_disagreement"
    weaker = pending.active_tasks[0]
    assert pending.best_metrics == {"quality": 0.8}
    assert weaker.mode == "probe" and weaker.metrics == {"quality": 0.4}
    assert candidate_parts(engine.evidence.get(weaker.code_artifact)["content"]["code"], engine.command)[0] == {"value": 2}
    engine.resolve_review(pending.pending_review, "TEST HUMAN ONLY", "Retain measured baseline", accept=False)
    engine.tick()  # Converge after test human decision.
    engine.record_refutation(weaker.evidence, "TEST HUMAN ONLY", "Weaker on the frozen validation fixture")
    assert refuted(engine, engine.evidence.get(weaker.code_artifact)["content"]["code"])
    engine.tick()  # Curate and request escape research.
    state = drive(engine, "monitor")
    task = state.active_tasks[0]
    assert task.hypothesis.machine_target == STRUCTURE_TARGET
    assert state.used == 2 and task.receipt
    crashed = subprocess.run([sys.executable, "-c", """
import os, signal, sys
from pathlib import Path
from loop_research.engine import Research
engine = Research(Path(sys.argv[1]))
engine.platform.poll = lambda *a, **kw: os.kill(os.getpid(), signal.SIGKILL)
engine.tick()
""", str(root)], timeout=12, check=False)
    assert crashed.returncode == -signal.SIGKILL and engine.store.read().lease_pid

    reopened = Research(root, worker=EscapeWorker(), radar=engine.radar)
    end = drive(reopened, "complete")
    assert end.used == 3 and end.offline_accepted
    assert end.active_tasks[0].receipt == task.receipt
    assert end.best_metrics == {"quality": 1}
    candidate = reopened.evidence.get(end.best_code)["content"]
    assert reopened.evidence.get(candidate["structural_review"])["content"]["review"]["accepted"]
    assert "candidate/model.py" in candidate["diff"] and "parameters.json" not in candidate["diff"]
    assert candidate_parts(candidate["code"], prediction_project.command_experiment) == ({"value": 1}, STRUCTURE)
    assert len([request for f in (root / "jobs").glob("*/request.json")
                if (request := json.loads(f.read_bytes()))["task_id"].startswith(task.id + ":")]) == 1
    assert reproduce(reopened, end.best_evidence)["passed"]
    assert reproduce(reopened, end.final_evidence)["passed"]
    held = reopened.evidence.get(end.final_evidence)["content"]
    assert held["evaluation_indices"] == ["z0", "z1"]


def test_structural_axis_rejects_parameter_and_whitespace_only_edits(tmp_path, prediction_project):
    engine = Research.initialize(tmp_path / "run", prediction_project)
    baseline = candidate_code({"value": 0}, BASE)
    with pytest.raises(ValueError, match="preserve all parameters"):
        engine.intervention(baseline, candidate_code({"value": 1}, STRUCTURE))
    with pytest.raises(ValueError, match="change the model"):
        engine.intervention(baseline, candidate_code({"value": 0}, BASE.replace("a, b", "a,b") + "# comment\n"))
