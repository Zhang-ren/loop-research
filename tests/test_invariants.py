import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_memory_lifecycle import MemoryReviewerDouble

from loop_research.engine import Research, axes_of
from loop_research.memory import Memory
from loop_research.models import MemoryDraft, Metric, Project, State
from loop_research.platform import LocalPlatform
from loop_research.storage import Conflict, Evidence, StateStore, atomic, canonical
from loop_research.training import BASELINE_CODE, measure, validate_code
from loop_research.workers import restricted_command


@pytest.fixture
def project():
    return Project.model_validate_json((Path(__file__).parents[1] / "examples/breast-cancer.json").read_bytes())


def test_missing_requirements_block_start():
    with pytest.raises(ValidationError):
        Project.model_validate({"name": "no-budget"})


@pytest.mark.parametrize("field", ["affordable", "reversible", "confirmed_knowledge"])
def test_scope_requires_confirmation(project, field):
    value = project.model_dump()
    value[field] = False
    with pytest.raises(ValidationError):
        Project.model_validate(value)


def test_metrics_all_must_pass():
    metrics = [Metric(name="a", threshold=0.9), Metric(name="b", threshold=0.95)]
    assert not all(m.passes({"a": 1.0, "b": 0.94}[m.name]) for m in metrics)
    assert not Metric(name="a", threshold=0.9).passes(float("nan"))


def test_state_compare_and_swap_and_recovery(tmp_path):
    store = StateStore(tmp_path)
    store.initialize(State(project_id="p"))
    store.change(lambda s: setattr(s, "batch", 1), expected=0)
    with pytest.raises(Conflict):
        store.change(lambda s: setattr(s, "batch", 2), expected=0)
    assert StateStore(tmp_path).read().batch == 1
    atomic(tmp_path / ".pending-interrupted", b"truncated")
    assert StateStore(tmp_path).read().revision == 1


def test_concurrent_state_writers(tmp_path):
    store = StateStore(tmp_path)
    store.initialize(State(project_id="p"))
    def increment(_):
        StateStore(tmp_path).change(lambda s: setattr(s, "used", s.used + 1))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(40)))
    assert store.read().used == 40


def test_evidence_chain_and_tampering(tmp_path):
    evidence = Evidence(tmp_path)
    a = evidence.put("raw", {"accuracy": 0.9})
    b = evidence.put("claim", {"statement": "measured"}, [a])
    assert evidence.verify_tree(b) == {a, b}
    (evidence.root / f"{a}.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        evidence.verify_tree(b)


@pytest.mark.parametrize("code", [
    "import os\ndef build_model(seed):\n return os.system('true')",
    "from sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return eval('1')",
    "from sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return LogisticRegression().fit([], [])",
    "from sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n open('/tmp/escape','w')\n return LogisticRegression()",
    "from sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return LogisticRegression(max_iter=100000000)",
    "from sklearn.linear_model import LogisticRegression as X\ndef build_model(seed):\n return X()",
    "from sklearn.linear_model import LogisticRegression\ndef build_model(seed=open('/tmp/a')):\n return LogisticRegression()",
    "from sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return LogisticRegression(**{})",
])
def test_model_cannot_replace_evaluator_or_do_io(code):
    with pytest.raises((ValueError, SyntaxError)):
        validate_code(code)


def test_structural_change_detected():
    scaled = BASELINE_CODE.replace("from sklearn.pipeline", "from sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline").replace("make_pipeline(LogisticRegression", "make_pipeline(StandardScaler(), LogisticRegression")
    assert axes_of(BASELINE_CODE)["classifier"] == axes_of(scaled)["classifier"]
    assert axes_of(BASELINE_CODE)["preprocessing"] != axes_of(scaled)["preprocessing"]


def test_memory_scope_approval_and_revocation(tmp_path):
    evidence = Evidence(tmp_path)
    raw = evidence.put("unverified_summary", {"metrics": {"a": 1}})
    memory = Memory(tmp_path, evidence)
    draft = MemoryDraft(title="scaling", content="scaling improved validation", evidence_ids=[raw],
                        applicability="this dataset", kind="skill")
    key = memory.propose(draft, "project-A", 1)
    assert memory.search("scaling", "project-A") == []
    with pytest.raises(ValueError, match="validation"):
        memory.approve(key, reviewer="test-human", expected_hash=key, explanation="fixture only")
    first = evidence.put("measured_baseline", measure(BASELINE_CODE, 42))
    second = evidence.put("training_result", measure(BASELINE_CODE.replace("C=1.0", "C=0.1"), 42))
    draft.evidence_ids = [first, second]
    key = memory.propose(draft, "project-A", 1)
    assert memory.propose(draft, "project-A", 1) == key
    with pytest.raises(ValueError):
        memory.approve(key, reviewer="test-human", expected_hash="wrong", explanation="fixture only")
    memory.review(key, MemoryReviewerDouble())
    memory.approve(key, reviewer="test-human", expected_hash=key, explanation="unit fixture, not a live approval")
    assert memory.search("scaling", "project-A")
    assert memory.search("scaling", "project-B") == []
    assert (memory.root / "skills" / memory.name(key) / "SKILL.md").exists()
    memory.revoke(key, "counterexample")
    assert not memory.search("scaling", "project-A")
    assert not (memory.root / "skills" / memory.name(key) / "SKILL.md").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="local sandbox adapter is macOS-only")
def test_real_sibling_read_and_write_isolation(tmp_path):
    workspace = tmp_path / "worker-A"
    workspace.mkdir()
    (workspace / "own").write_text("own")
    (tmp_path / "sibling").write_text("not authorized")
    for target, success in [(workspace / "own", True), (tmp_path / "sibling", False)]:
        run = subprocess.run(restricted_command(["/bin/cat", str(target)], tmp_path, workspace), capture_output=True)
        assert (run.returncode == 0) is success
    run = subprocess.run(restricted_command([sys.executable, "-c", "from pathlib import Path; Path('" + str(tmp_path / "sibling") + "').write_text('overwrite')"], tmp_path, workspace), capture_output=True)
    assert run.returncode != 0
    assert (tmp_path / "sibling").read_text() == "not authorized"


@pytest.mark.skipif(sys.platform != "darwin", reason="local sandbox adapter is macOS-only")
def test_runner_cannot_modify_frozen_code_outside_worker(tmp_path):
    workspace = tmp_path / "research" / "worker-A"
    workspace.mkdir(parents=True)
    frozen = tmp_path / "frozen-evaluator.py"
    frozen.write_text("frozen")
    command = [sys.executable, "-c", f"from pathlib import Path; Path({str(frozen)!r}).write_text('changed')"]
    run = subprocess.run(restricted_command(command, workspace.parent, workspace), capture_output=True, check=False)
    assert run.returncode != 0
    assert frozen.read_text() == "frozen"


def test_detached_training_idempotency_and_metric_validation(tmp_path):
    platform = LocalPlatform(tmp_path)
    key = platform.submit("same-task", BASELINE_CODE, 42)
    assert platform.submit("same-task", BASELINE_CODE, 42) == key
    assert len(list(platform.root.glob("*/receipt.json"))) == 1
    deadline = time.time() + 40
    while time.time() < deadline:
        result = LocalPlatform(tmp_path).poll(key, ["accuracy", "roc_auc"])
        if result["status"] != "running":
            break
        time.sleep(0.2)
    assert result["status"] == "succeeded"
    assert 0 <= result["metrics"]["accuracy"] <= 1
    result["metrics"].pop("accuracy")
    atomic(platform.root / key / "result.json", canonical(result))
    with pytest.raises(ValueError, match="Missing"):
        platform.poll(key, ["accuracy"])


def test_unknown_local_submission_is_reconciled_without_duplicate_training(tmp_path):
    import json
    platform = LocalPlatform(tmp_path)
    key = platform.submit("recover-receipt", BASELINE_CODE, 42)
    path = platform.root / key / "receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt.update(pid=None)
    atomic(path, canonical(receipt))
    result = platform.reconcile(key, ["accuracy"])
    assert result["status"] in {"running", "succeeded"}
    assert len(list(platform.root.glob("*/request.json"))) == 1
    # Ensure the detached job completes before the temporary test directory is removed.
    deadline = time.time() + 40
    while platform.poll(key)["status"] == "running" and time.time() < deadline:
        time.sleep(0.1)
    assert platform.poll(key)["status"] == "succeeded"


def test_stop_invalidates_late_results_and_prevents_dispatch(tmp_path, project):
    # Avoid re-running baseline/MLflow: construct the exact persisted state contract directly.
    atomic(tmp_path / "project.json", canonical(project.model_dump()))
    store = StateStore(tmp_path)
    store.initialize(State(project_id="p", lease="inflight", lease_until=time.time()+600))
    engine = Research(tmp_path)
    engine.lease = "inflight"
    engine.stop()
    with pytest.raises(Conflict, match="Late"):
        engine.update(lambda s: setattr(s, "used", 1), "late")
    assert store.read().used == 0
    assert store.read().phase == "stopped"


def test_measured_data_split_has_no_train_evaluation_overlap():
    result = measure(BASELINE_CODE, 42)
    assert not set(result["train_indices"]) & set(result["evaluation_indices"])
    assert len(result["labels"]) == len(result["predictions"]) == len(result["scores"])
    assert result["split"] == "validation"
