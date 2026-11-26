"""Run the isolated-model acceptance case on the previously frozen public digits split."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

from run_public_paper_study import StudyRadar, copy_evidence, raw_source, read

from loop_research.command_candidate import STRUCTURE_TARGET, candidate_parts
from loop_research.engine import Research
from loop_research.models import Project, StrictModel
from loop_research.preflight import inspect
from loop_research.replay import reproduce
from loop_research.storage import Evidence, atomic, canonical, digest


MODEL = """from sklearn.svm import SVC

def build_model(seed, parameters):
    return SVC(C=parameters["C"], gamma=0.001, kernel="rbf")
"""
PREDICT = """import json
import sys
import time
from pathlib import Path
from model import build_model

data = json.loads(Path(sys.argv[1]).read_text())
assert set(data) == {"train", "features", "parameters", "seed"}
assert not Path("samples.json").exists() and not Path("score.py").exists()
model = build_model(data["seed"], data["parameters"])
model.fit([row["features"] for row in data["train"]], [row["label"] for row in data["train"]])
predictions = [int(value) for value in model.predict(data["features"])]
time.sleep(3)  # Reserve a window for the controlled monitor crash.
Path(sys.argv[2]).write_text(json.dumps({"predictions": predictions}))
"""
SCORE = """import json
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

data = json.loads(Path(sys.argv[1]).read_text())
assert set(data) == {"labels", "predictions"} and not Path("model.py").exists()
y, pred = data["labels"], data["predictions"]
metrics = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
Path(sys.argv[2]).write_text(json.dumps({"metrics": metrics}))
"""


class ReuseDecision(StrictModel):
    decision: Literal["measure", "skip"]
    memory_id: str | None
    reason: str


def compare_memory(engine, proposal, other_run=None):
    """Test a fresh decision with and without one real, approved study memory."""
    batch = read(engine.root / "batches/000001.json")["evidence"]
    first = engine.evidence.get(batch)["content"]["tasks"][0]
    code = engine.evidence.get(first["code_artifact"])["content"]["code"]
    parameters, _ = candidate_parts(code, engine.command)
    draft = engine.memory.read(proposal)
    output = engine.root / ("memory-comparison-" + proposal + ".json")
    approved_path = engine.memory.root / "approved" / (proposal + ".json")
    if not approved_path.exists():
        atomic(output, canonical({"status": "awaiting_human_approval", "proposal": proposal, "batch": batch}))
        return 3
    scope = engine.store.read().project_id
    records = engine.memory.search(engine.project.objective, scope)
    approved = next((record for record in records if record["id"] == proposal), None)
    if not approved or first["evidence"] not in draft["draft"]["evidence_ids"]:
        raise ValueError("Memory comparison requires an active, retrieved memory citing the first candidate measurement")
    payload = {"objective": engine.project.objective, "candidate_parameters": parameters,
               "experiment_context": engine.project.knowledge, "project_scope": scope,
               "protocol": engine.references()["protocol"],
               "requirements": "Given only this handoff, decide whether to measure this exact configuration. "
               "Skip it only if an approved memory supports an already measured degradation under the same frozen protocol. "
               "If no matching approved memory is supplied, measure the candidate. Cite its exact memory ID when skipping. "
               "Do not run tools or claim a training job has happened."}
    decisions = {}
    for name, memory in (("without", []), ("with", [approved])):
        result, meta = engine.worker.run("memory-reuse-audit", {**payload, "approved_memory": memory}, ReuseDecision)
        decisions[name] = {"decision": result.model_dump(), "worker": meta}
    old, new = (decisions[name]["decision"] for name in ("without", "with"))
    passed = old["decision"] == "measure" and old["memory_id"] is None and new["decision"] == "skip" and new["memory_id"] == proposal
    other_check = None
    if other_run:
        other = Research(other_run)
        other_scope = other.store.read().project_id
        retrieved = other.memory.search(engine.project.objective, other_scope)
        isolated = other_scope != scope and proposal not in {item["id"] for item in retrieved}
        other_check = {"scope": other_scope, "approved_memory_absent": isolated}
        passed = passed and isolated
    outcome = {"status": "passed" if passed else "no_decision_difference", "proposal": proposal,
               "decisions": decisions, "other_project": other_check,
               "no_candidate_dispatched": True, "approval_sha256": digest(approved_path.read_bytes())}
    key = engine.evidence.put("approved_memory_decision_comparison", outcome, [batch, *draft["draft"]["evidence_ids"]])
    atomic(output, canonical({**outcome, "evidence": key}))
    return 0 if passed else 2


def current_engine(root: Path) -> Research:
    engine = Research(root)
    engine.radar = StudyRadar(root, engine.evidence)
    return engine


def initialize(root, interpreter, public_run):
    source = root.parent / (root.name + "-source")
    source.mkdir(parents=True, exist_ok=True)
    split = public_run / "experiment/split.json"
    atomic(source / "split.json", split.read_bytes())
    samples = subprocess.check_output([str(interpreter), "-I", "-c", """
import json
from sklearn.datasets import load_digits
x, y = load_digits(return_X_y=True)
print(json.dumps({str(i): {"features": a.astype(int).tolist(), "label": int(b)}
                  for i, (a, b) in enumerate(zip(x, y))}))
"""], timeout=30)
    atomic(source / "samples.json", samples)
    for name, code in (("model.py", MODEL), ("predict.py", PREDICT), ("score.py", SCORE)):
        atomic(source / name, code.encode())
    git = ["git", "-C", str(source)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    if subprocess.check_output([*git, "diff", "--cached", "--name-only"]):
        subprocess.run([*git, "-c", "user.name=LOOP Acceptance", "-c", "user.email=local@invalid",
                        "commit", "-qm", "Freeze public digits inputs and evaluator"], check=True)
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    project = Project.model_validate({
        "name": "Public digits structural-search acceptance",
        "objective": "Improve public 8 by 8 digits validation accuracy and macro F1 under a frozen split and scorer.",
        "knowledge": "The SVC control uses raw integer pixel intensities, C=1, gamma=0.001, and the RBF kernel. "
                     "The initial pilot varies only C. A single validation decrease can trigger a structurally different "
                     "feature or model hypothesis. Only the trusted scorer receives the evaluation labels. "
                     "All candidate comparisons use the existing public split and both metrics. "
                     "Installed experiment dependencies are sklearn, numpy and scipy.",
        "archive_uri": "local", "metrics": [{"name": key, "threshold": 1} for key in ("accuracy", "macro_f1")],
        "max_experiments": 2, "branches": 1, "max_batches": 2, "stagnation_batches": 1,
        "independent_review": True, "reversible": True, "affordable": True, "confirmed_knowledge": True,
        "command_experiment": {"repository": str(source), "revision": revision, "command": ["isolated-predict"],
                               "split_manifest": "split.json", "bridge_interpreter": str(interpreter),
                               "baseline": {"C": 1.0}, "choices": {"C": [1.0, 0.01]}, "timeout_seconds": 40},
    })
    result = inspect(project)
    if result["issues"]:
        raise ValueError(result["issues"])
    atomic(root / "acceptance-project.json", canonical(project.model_dump()))
    engine = Research.initialize(root, project)
    install_evidence(root, public_run, engine)


def install_evidence(root, public_run, engine):
    pointer = root / "literature/admitted-sources.json"
    if pointer.exists():
        engine.evidence.verify_tree(read(pointer)["sources"][0])
        return
    discovery = read(public_run / "discovery.json")
    source = Evidence(public_run)
    copy_evidence(source, engine.evidence, [discovery["evidence"]])
    relative = "doc/modules/svm.rst"
    text = raw_source(discovery["selection"]["repository"], discovery["commit"], relative)
    start = text.index(b"* **Setting C**:")
    end = text.index(b".. _shrinking_svm:", start)
    excerpt = text[start:end].decode().strip()
    content = {"title": "scikit-learn SVM practical guidance on regularization and input scaling",
               "access": "body_excerpt", "body_excerpt": excerpt, "abstract": "", "retracted": False,
               "url": "https://scikit-learn.org/1.6/modules/svm.html#tips-on-practical-use",
               "source_sha256": digest(text), "source_commit": discovery["commit"],
               "kind": "official implementation documentation"}
    key = engine.evidence.put("paper_source", content, [discovery["evidence"]])
    atomic(pointer, canonical({"sources": [key], "discovery": discovery["evidence"]}))


def crash_monitor(engine, state):
    receipt = state.active_tasks[0].receipt
    if not receipt or (engine.root / "controller-recovery.json").exists():
        return
    status = engine.platform.poll(receipt)["status"]
    if status != "running":
        return
    trigger = subprocess.run([sys.executable, __file__, str(engine.root), "--crash-monitor"],
                             timeout=20, check=False)
    if trigger.returncode != -signal.SIGKILL:
        raise RuntimeError(f"Crash monitor returned {trigger.returncode}")
    restarted = current_engine(engine.root).tick()
    if restarted.active_tasks[0].receipt != receipt:
        raise RuntimeError("Controller recovery changed the candidate receipt")
    atomic(engine.root / "controller-recovery.json", canonical({
        "candidate_receipt": receipt, "status_before_crash": status, "crash_code": trigger.returncode,
        "phase_after_recovery": restarted.phase, "preserved": True
    }))


def audit(engine):
    state = engine.store.read()
    task = state.active_tasks[0]
    code = engine.evidence.get(task.code_artifact)["content"]
    first_batch = read(engine.root / "batches/000001.json")["evidence"]
    early = engine.evidence.get(first_batch)["content"]["tasks"][0]
    recovery = read(engine.root / "controller-recovery.json")
    validation = engine.evidence.get(task.evidence)["content"]
    holdout = engine.evidence.get(state.final_evidence)["content"]
    requests = [read(path) for path in (engine.root / "jobs").glob("*/request.json")]
    validation_jobs = [request for request in requests if request["task_id"].startswith(("b1-", "b2-"))]
    measured = {task.evidence, state.final_evidence}
    reproductions = {}
    for path in sorted(engine.evidence.root.glob("*.json")):
        record = engine.evidence.get(path.stem)
        content = record["content"]
        if record["kind"] == "reproduction" and content["original"] in measured and content["passed"]:
            engine.evidence.verify_tree(path.stem)
            reproductions[content["original"]] = path.stem
    for key in sorted(measured - reproductions.keys()):
        replay = reproduce(engine, key)
        if replay["passed"]:
            reproductions[key] = replay["evidence"]
    passes = (
        task.hypothesis.machine_target == STRUCTURE_TARGET
        and candidate_parts(code["code"], engine.command)[0] == engine.command.baseline
        and engine.evidence.get(code["structural_review"])["content"]["review"]["accepted"]
        and state.used == len(validation_jobs) == 2
        and recovery["preserved"] and recovery["status_before_crash"] == "running"
        and not set(holdout["evaluation_indices"]) & set(validation["evaluation_indices"])
        and set(reproductions) == measured
    )
    result = {"passed": passes, "batches": state.completed_batches, "candidate_tasks": state.used,
              "baseline_metrics": engine.evidence.get(state.baseline_evidence)["content"]["metrics"],
              "first_candidate": early["metrics"], "selected_validation_metrics": state.best_metrics,
              "new_structure_metrics": task.metrics, "holdout_metrics": holdout["metrics"],
              "diff": code["diff"], "structure_evidence": task.evidence,
              "holdout_evidence": state.final_evidence, "controller_recovery": recovery,
              "reproductions": reproductions}
    atomic(engine.root / "structural-audit.json", canonical(result))
    return 0 if passes else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interpreter", type=Path)
    parser.add_argument("--public-run", type=Path)
    parser.add_argument("--pause-first-round", action="store_true")
    parser.add_argument("--compare-memory", metavar="APPROVED_HASH")
    parser.add_argument("--check-other-run", type=Path)
    parser.add_argument("--crash-monitor", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.crash_monitor:
        engine = current_engine(root)
        if engine.store.read().phase != "monitor":
            return 2
        engine.platform.poll = lambda *args, **kwargs: os.kill(os.getpid(), signal.SIGKILL)
        engine.tick()
        return 2
    if args.compare_memory:
        return compare_memory(current_engine(root), args.compare_memory, args.check_other_run)
    if not (root / "state.json").exists():
        if not args.interpreter or not args.public_run:
            parser.error("new acceptance run needs --public-run and --interpreter")
        initialize(root, args.interpreter.resolve(), args.public_run.resolve())
    elif args.public_run:
        install_evidence(root, args.public_run.resolve(), current_engine(root))
    for _ in range(100):
        engine = current_engine(root)
        state = engine.store.read()
        if state.phase == "complete":
            return audit(engine)
        if state.phase in {"review", "stopped"}:
            print(f"{state.phase}: {state.stop_reason}", flush=True)
            return 3
        if args.pause_first_round and state.phase == "research" and state.completed_batches == 1:
            return 0
        state = engine.tick()
        print(f"phase={state.phase} research={state.research_mode}/{state.research_stage} "
              f"batch={state.batch} used={state.used}", flush=True)
        if state.phase == "monitor":
            crash_monitor(engine, state)
        time.sleep(0.25)
    return 3


if __name__ == "__main__":
    sys.exit(main())
