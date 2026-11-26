"""Real local command execution with labeled research and reviewer fixtures."""
import json
import shutil
import subprocess
import sys
import time

import pytest
from test_memory_lifecycle import MemoryReviewerDouble
from test_workflow import RadarDouble, WorkerDouble

from loop_research.engine import Research
from loop_research.followup import refuted
from loop_research.models import CodeResult, Hypothesis, IndependentReview, MemoryDraft, Plan, Project, TradeoffPlan
from loop_research.replay import reproduce

PROGRAM = """import json
import os
from pathlib import Path

params = json.loads(Path(os.environ['LOOP_PARAMETERS']).read_text())
splits = json.loads(Path(os.environ['LOOP_SPLIT_MANIFEST']).read_text())
split = os.environ['LOOP_SPLIT']
assert set(splits) == {'train', split}
assert not Path('splits.json').exists()
assert not set(splits['train']).intersection(splits[split])
quality = 10 + params['alpha'] + params['beta'] - params['gamma']
Path(os.environ['LOOP_METRICS']).write_text(json.dumps({'metrics': {'quality': quality}}))
"""


class CommandWorker(WorkerDouble):
    disagree = False

    def run(self, role, payload, response, **kwargs):
        meta = {"TEST_DOUBLE": True, "role": role}
        if role == "planner":
            self.calls.append(role)
            return Plan(rationale="TEST planned configuration", hypotheses=[
                Hypothesis(title=entry["title"], question="Does the setting help?", axis=entry["machine_target"],
                           change="test next option", prediction="measure the impact", falsification="no gain",
                           source_ids=payload["source_ids"], parameter_bounds="declared grid", single_variable=True,
                           machine_target=entry["machine_target"], direction_id=entry["id"])
                for entry in payload["direction_pool"][:payload["branches"]]
            ]), meta
        if role == "coder":
            self.calls.append(role)
            name = payload["hypothesis"]["machine_target"].split(".")[-1]
            values = json.loads(payload["baseline_code"])
            values[name] = next(value for value in payload["parameter_choices"][name] if value != values[name])
            return CodeResult(code=json.dumps(values), rationale="TEST parameter change",
                              source_ids=payload["source_ids"], changed_axis=name), meta
        if role == "reviewer":
            self.calls.append(role)
            assert "previous_rejection" not in payload
            best = max(payload["scores"], key=payload["scores"].get)
            return IndependentReview(selected_evidence=None if self.disagree else best,
                                     evidence_ids=payload["evidence_ids"], rationale="TEST blind decision",
                                     concerns=[]), meta
        return super().run(role, payload, response, **kwargs)


@pytest.fixture
def setup_project(tmp_path):
    repo = tmp_path / "existing"
    repo.mkdir()
    (repo / "evaluate.py").write_text(PROGRAM)
    (repo / "splits.json").write_text(json.dumps({"train": ["a", "b"], "validation": ["c"], "test": ["d"]}))
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "Existing experiment"], check=True)
    project = Project.model_validate({
        "name": "Independent test repository", "objective": "Improve quality of the existing non-sklearn command",
        "knowledge": "TEST FIXTURE experiment with a fixed split manifest and declared parameters.",
        "archive_uri": "local", "metrics": [{"name": "quality", "threshold": 12}], "max_experiments": 4,
        "reversible": True, "affordable": True, "confirmed_knowledge": True,
        "independent_review": True, "followup_experiments": True,
        "command_experiment": {"repository": str(repo), "command": [sys.executable, "evaluate.py"],
                               "split_manifest": "splits.json", "baseline": {"alpha": 0, "beta": 0, "gamma": 0},
                               "choices": {"alpha": [0, 1], "beta": [0, 1], "gamma": [0, 1]}},
    })
    return repo, project


def drive(engine, phases):
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        state = engine.tick()
        if state.phase in phases:
            return state
        time.sleep(0.04)
    raise AssertionError(engine.store.read())


def test_first_metric_plan_does_not_cite_missing_history(tmp_path, setup_project):
    _, original = setup_project
    data = original.model_dump()
    data["metric_policy"] = {"targets": data["metrics"], "hard_limits": []}
    engine = Research.initialize(tmp_path / "policy-run", Project.model_validate(data))

    class MetricWorker(CommandWorker):
        def run(self, role, payload, response, **kwargs):
            if role == "planner":
                assert "previous_analysis_id" not in payload
                assert all(isinstance(key, str) for key in payload["source_ids"])
            return super().run(role, payload, response, **kwargs)

    engine.worker = MetricWorker()
    engine.radar = RadarDouble(engine.evidence)
    state = drive(engine, {"execute"})
    assert state.batch == 1 and state.active_plan
    assert state.last_analysis is None
    engine.evidence.verify_tree(state.active_plan)


def test_declared_controls_resume_after_research_worker_timeout(tmp_path, setup_project):
    from loop_research.workers import WorkerFailure

    _, project = setup_project
    engine = Research.initialize(tmp_path / "research-timeout", project)

    class TimedOutWorker:
        def run(self, role, payload, response, **kwargs):
            assert role == "research-query"
            raise WorkerFailure("TEST tool timeout")

    engine.worker = TimedOutWorker()
    assert engine.tick().phase == "research"
    state = engine.tick()
    assert state.phase == "research" and state.research_failures == 1
    state = engine.tick()
    assert state.phase == "plan" and state.used == 0
    pool = engine.evidence.get(state.direction_pool)["content"]
    assert pool["repository_controls"] is True
    assert {item["machine_target"] for item in pool["directions"]} == {
        "parameter.alpha", "parameter.beta", "parameter.gamma",
    }


def test_complete_plan_retains_future_questions_without_restarting_retrieval(tmp_path, setup_project):
    _, project = setup_project
    engine = Research.initialize(tmp_path / "later-questions", project)

    class LaterQuestionWorker(CommandWorker):
        def run(self, role, payload, response, **kwargs):
            result, receipt = super().run(role, payload, response, **kwargs)
            if role == "planner":
                result.research_questions = ["Will the resulting change generalize beyond the frozen split?"]
            return result, receipt

    engine.worker = LaterQuestionWorker()
    engine.radar = RadarDouble(engine.evidence)
    state = drive(engine, {"execute"})
    plan = engine.evidence.get(state.active_plan)["content"]["plan"]
    assert plan["research_questions"] and len(state.active_tasks) == project.branches


def test_opposing_measured_objectives_must_be_carried_into_next_plan(tmp_path, setup_project):
    repo, original = setup_project
    scorer = repo / "evaluate.py"
    scorer.write_text(PROGRAM.replace("'quality': quality", "'quality': quality, 'cost': params['alpha'] + params['beta']"))
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qam", "Fixture adds measured cost"], check=True)
    targets = [{"name": "quality", "threshold": 14}, {"name": "cost", "direction": "min", "threshold": 0}]
    policy = {"targets": targets, "hard_limits": [{"name": "cost", "direction": "min", "threshold": 2}]}
    data = {**original.model_dump(), "metrics": targets, "metric_policy": policy, "independent_review": False}
    engine = Research.initialize(tmp_path / "tradeoff-run", Project.model_validate(data))

    class MetricWorker(CommandWorker):
        tradeoff_calls = 0

        def run(self, role, payload, response, **kwargs):
            plan, meta = super().run(role, payload, response, **kwargs)
            if role != "planner" or "opposing_measurements" not in payload:
                return plan, meta
            self.tradeoff_calls += 1
            assert response is TradeoffPlan
            assert payload["required_protected_metrics"] == ["cost"]
            assert all(row["better"] == ["quality"] and row["worse"] == ["cost"]
                       for row in payload["opposing_measurements"].values())
            evidence_ids = payload["evidence_ids"]
            return response(**plan.model_dump(), tradeoff={
                "evidence_ids": evidence_ids[:-1] if self.tradeoff_calls == 1 else evidence_ids,
                "protected_metrics": ["cost"],
                "relationship": "Fixture quality improved, and the corresponding cost increased.",
                "next_check": "Measure another allowed setting against the fixed quality and cost policy.",
            }), meta

    engine.worker = MetricWorker()
    engine.radar = RadarDouble(engine.evidence)
    assert drive(engine, {"execute"}).batch == 1
    state = drive(engine, {"execute", "review", "complete"})
    assert state.phase == "execute" and state.batch == 2 and engine.worker.tradeoff_calls == 2
    approved_plan = engine.evidence.get(state.active_plan)
    followup = approved_plan["content"]["plan"]["tradeoff"]
    assert set(followup["evidence_ids"]) <= set(approved_plan["parents"])
    assert set(followup["evidence_ids"]) == {state.last_analysis, *engine.opposing_results(state.last_analysis)}
    assert any(engine.evidence.get(path.stem)["kind"] == "rejected_plan"
               for path in engine.evidence.root.glob("*.json"))


@pytest.mark.parametrize("verdict", ["supported", "inconclusive"])
def test_direction_refutation_uses_measured_conditions_and_explicit_review(tmp_path, setup_project, verdict):
    from loop_research.command_candidate import candidate_code
    from loop_research.platform import measure_command
    from loop_research.refutations import active_rule, decide, propose
    from loop_research.storage import atomic, canonical

    _, config = setup_project
    engine = Research.initialize(tmp_path / "scoped", config)
    protocol = engine.references()["protocol"]
    measurements = []
    def measure(alpha, gamma):
        code = candidate_code({"alpha": alpha, "beta": 0, "gamma": gamma}, None)
        artifact = engine.evidence.put("model_code", {"code": code}, [protocol])
        result = measure_command(engine.root, engine.command, engine.platform.source, code, config.seed,
                                 "validation", engine.root / f"fixture-{alpha}-{gamma}")
        assert result["metrics"]["quality"] == 10 + alpha - gamma
        return engine.evidence.put("training_result", {**result, "status": "succeeded"}, [artifact, protocol])

    for batch, alpha in enumerate((0, 1), 1):
        baseline, candidate = measure(alpha, 0), measure(alpha, 1)
        measurements.append(candidate)
        snapshot = engine.evidence.put("batch_snapshot", {"batch": batch, "baseline": baseline,
            "tasks": [{"evidence": candidate, "status": "succeeded", "hypothesis": {"machine_target": "parameter.gamma"}}]},
            [baseline, candidate])
        decision = engine.evidence.put("batch_decision", {"snapshot": snapshot}, [snapshot])
        atomic(engine.root / "batches" / f"{batch:06d}.json", canonical({"evidence": snapshot}))
        atomic(engine.root / "decisions" / f"{batch:06d}.json", canonical({"evidence": decision}))

    class Assessor:
        def run(self, role, payload, response):
            assert role == "refutation-attribution-review" and len(payload["paired_measurements"]) == 2
            return response(verdict=verdict, explanation="Controlled comparisons with separate matched baselines.",
                            limitations=["The recorded cases only"], evidence_ids=payload["evidence_ids"]), {}

    engine.reviewer = Assessor()
    queue = propose(engine, measurements)
    variant = candidate_code({"alpha": 0, "beta": 0, "gamma": 1}, None)
    assert queue["status"] == "awaiting_human" and active_rule(engine, variant) is None
    if verdict == "inconclusive":
        with pytest.raises(ValueError, match="remains pending"):
            decide(engine, queue["proposal"], "fixture", "Check the evidence", "approve")
        result = decide(engine, queue["proposal"], "fixture", "Wait for more evidence", "defer")
        assert result["status"] == "awaiting_human" and active_rule(engine, variant) is None
        return
    result = decide(engine, queue["proposal"], "fixture", "Approve only the measured contexts", "approve")
    assert active_rule(engine, variant) == result["decision"]
    assert active_rule(engine, candidate_code({"alpha": 1, "beta": 0, "gamma": 1}, None)) == result["decision"]
    assert active_rule(engine, candidate_code({"alpha": 0, "beta": 1, "gamma": 1}, None)) is None
    engine.schedule_followup(engine.store.read(), "probe", variant, [protocol], {})
    state = engine.tick()
    assert state.active_tasks[0].status == "cancelled" and state.active_tasks[0].receipt is None
    assert not list((engine.root / "jobs").glob("*/request.json"))


def test_existing_repo_combination_review_resume_and_replay(tmp_path, setup_project):
    repo, project = setup_project
    engine = Research.initialize(tmp_path / "run", project)
    original = engine.store.read()
    assert original.best_metrics == {"quality": 10}
    # A dirty edit after initialization cannot change the pinned baseline.
    (repo / "evaluate.py").write_text("raise RuntimeError('dirty input must be ignored')")
    worker = CommandWorker()
    worker.disagree = True
    engine.worker = engine.reviewer = worker
    engine.radar = RadarDouble(engine.evidence)
    pending = drive(engine, {"review"})
    assert pending.stop_reason == "independent_review_disagreement"
    assert pending.used == 3
    assert pending.best_evidence == original.best_evidence
    vote = engine.evidence.get(pending.pending_review)["content"]
    assert "low_confidence" in vote["triggers"] and "direction_decision" in vote["triggers"]
    calls = worker.calls.count("reviewer")
    engine.resolve_review(pending.pending_review, "TEST HUMAN ONLY", "Review fixture decision", accept=True)
    assert engine.tick().phase == "curate"
    assert worker.calls.count("reviewer") == calls
    weak = next(t for t in pending.active_tasks if t.metrics["quality"] < 10)
    decision = engine.record_refutation(weak.evidence, "TEST HUMAN ONLY", "Measured degradation in this fixture")
    assert engine.evidence.get(decision)["kind"] == "scoped_refutation"
    assert refuted(engine, engine.evidence.get(weak.code_artifact)["content"]["code"])
    new = Research(engine.root, worker=CommandWorker(), radar=engine.radar)
    end = drive(new, {"complete", "review"})
    assert end.phase == "complete" and end.offline_accepted and end.used == 4
    assert end.active_tasks[0].mode == "combination"
    assert new.evidence.get(end.final_evidence)["content"]["metrics"] == {"quality": 12}
    review_ref = json.loads((new.root / "reviews/000002-replay.json").read_bytes())["evidence"]
    assert new.evidence.get(review_ref)["content"]["matched"]
    assert reproduce(new, end.final_evidence)["passed"]
    assert (repo / "evaluate.py").read_text().startswith("raise RuntimeError")


def test_distinct_research_roots_reject_copied_project_memory(tmp_path, setup_project):
    _, project = setup_project
    one = Research.initialize(tmp_path / "one", project)
    two = Research.initialize(tmp_path / "two", project)
    a, b = one.store.read(), two.store.read()
    assert a.project_id != b.project_id
    evidence = [a.baseline_evidence]
    evidence.append(one.evidence.put("tool_observation", {"invocation_id": "fixture", "raw_sha256": "fixture",
                        "outcome": "baseline measured", "verified_by_controller": True}))
    draft = MemoryDraft(kind="skill", title="Project-specific baseline", content="Only repeat within this frozen experiment.",
                        applicability="TEST project A only", evidence_ids=evidence)
    key = one.memory.propose(draft, a.project_id, 1)
    assert one.memory.timeline(a.project_id)[0]["id"] == key
    assert two.memory.timeline(b.project_id) == []
    assert one.memory.search("", a.project_id) == []
    one.memory.review(key, MemoryReviewerDouble())
    one.memory.approve(key, reviewer="TEST HUMAN ONLY", expected_hash=key, explanation="Fixture approval")
    assert one.memory.search("", a.project_id)[0]["id"] == key
    assert two.memory.search("", b.project_id) == []
    for area in ("approved", "candidates", "skills"):
        shutil.copytree(one.memory.root / area, two.memory.root / area)
    with pytest.raises(ValueError, match="different research project"):
        two.memory.read(key, "approved")
    with pytest.raises(ValueError, match="different research project"):
        two.memory.search("", a.project_id)


def test_batch_budget_counts_convergence_separately_from_jobs(tmp_path, setup_project):
    _, project = setup_project
    project.max_batches = 1
    project.independent_review = False
    engine = Research.initialize(tmp_path / "batch-budget", project)
    engine.worker = CommandWorker()
    engine.radar = RadarDouble(engine.evidence)
    state = drive(engine, {"complete", "review"})
    assert state.phase == "complete" and state.completed_batches == 1
    assert state.used == 3 < project.max_experiments
    decision = json.loads((engine.root / "decisions/000001.json").read_bytes())["evidence"]
    record = engine.evidence.get(decision)
    assert record["kind"] == "batch_decision" and record["content"]["selected"] == state.best_evidence
    assert state.final_receipt is not None and not state.offline_accepted


def test_dead_lease_is_recovered(tmp_path, setup_project):
    _, project = setup_project
    engine = Research.initialize(tmp_path / "run", project)
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = process.pid
    process.wait()
    def stale(state):
        state.lease = "previous-owner"
        state.lease_until = time.time() + 1000
        state.lease_pid, state.lease_process_created = pid, 1
    engine.store.change(stale)
    assert engine.tick().phase == "research"


def test_boundary_probe_runs_as_measured_followup(tmp_path, setup_project):
    _, config = setup_project
    value = config.model_dump()
    value.update(branches=1, max_experiments=2, independent_review=False)
    value["metrics"][0]["threshold"] = 100  # Leave headroom for a second experiment.
    value["command_experiment"]["choices"]["alpha"] = [0, 1, 3]
    engine = Research.initialize(tmp_path / "probe", Project.model_validate(value))
    engine.worker = CommandWorker()
    engine.radar = RadarDouble(engine.evidence)
    end = drive(engine, {"complete"})
    assert end.used == 2
    assert end.active_tasks[0].mode == "probe"
    plan = engine.evidence.get(end.active_plan)["content"]
    assert plan["details"] == {"parameter": "alpha", "declared_range": [0, 3],
                               "local_range": [0, 3], "tested_value": 3}
    measured = engine.evidence.get(end.active_tasks[0].evidence)["content"]
    assert measured["metrics"] == {"quality": 13}
    assert not end.offline_accepted


def test_lost_submit_ack_reuses_frozen_candidate_and_job(tmp_path, setup_project):
    _, project = setup_project
    engine = Research.initialize(tmp_path / "run", project)
    worker = CommandWorker()
    engine.worker, engine.radar = worker, RadarDouble(engine.evidence)
    drive(engine, {"execute"})
    original = engine.platform.submit

    def lost_ack(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("TEST FIXTURE lost acknowledgement after actual submission")

    engine.platform.submit = lost_ack
    with pytest.raises(RuntimeError, match="lost acknowledgement"):
        engine.tick()
    assert len(list(engine.platform.root.glob("*/request.json"))) == 1
    requests = {p.parent.name for p in engine.platform.root.glob("*/request.json")}
    new = Research(engine.root, worker=worker, radar=engine.radar)
    assert new.tick().phase == "monitor"
    jobs = {p.parent.name for p in new.platform.root.glob("*/request.json")}
    assert requests.issubset(jobs) and len(jobs) == 3
    assert worker.calls.count("coder") == 5  # The admitted candidate was preserved without another call.
    drive(new, {"converge"})


def test_infrastructure_retries_have_distinct_receipts_and_no_experiment_charge(tmp_path, setup_project):
    _, project = setup_project
    project.branches = 1
    engine = Research.initialize(tmp_path / "retry", project)
    engine.worker = CommandWorker()
    engine.radar = RadarDouble(engine.evidence)
    submitted = []
    failures = set()
    api_submit, api_poll = engine.platform.submit, engine.platform.poll

    def submit(task_id, code, seed, **kwargs):
        attempt = int(task_id.rsplit(":", 1)[-1])
        submitted.append(attempt)
        if attempt < 2:
            receipt = f"TEST-INFRASTRUCTURE-{attempt}"
            failures.add(receipt)
            return receipt
        return api_submit(task_id, code, seed, **kwargs)

    def poll(receipt, metrics=None):
        if receipt in failures:
            return {"status": "infra_failed", "error": "TEST simulated worker loss"}
        return api_poll(receipt, metrics)

    engine.platform.submit, engine.platform.poll = submit, poll
    state = drive(engine, {"converge", "review"})
    assert state.phase == "converge" and state.used == 1
    assert submitted == [0, 1, 2] and state.active_tasks[0].attempts == 3
    assert engine.worker.calls.count("coder") == 1


def test_evidence_bound_refutation_blocks_resubmission(tmp_path, setup_project):
    _, project = setup_project
    project.independent_review = False
    engine = Research.initialize(tmp_path / "refutation", project)
    engine.worker = CommandWorker()
    engine.radar = RadarDouble(engine.evidence)
    state = drive(engine, {"curate"})
    weak = next(task for task in state.active_tasks if task.metrics["quality"] < 10)
    decision = engine.record_refutation(weak.evidence, "TEST FIXTURE", "Regression on the same frozen fixture")
    previous_jobs = set(engine.platform.root.glob("*/request.json"))

    def reintroduce(current):
        current.phase = "execute"
        current.active_tasks = [weak.model_copy(update={"id": "reintroduced-fixture", "status": "reserved",
                                                       "attempts": 0, "counted": False, "receipt": None})]

    engine.store.change(reintroduce, event="TEST_reintroduce_refuted_candidate")
    blocked = engine.tick()
    assert blocked.active_tasks[0].status == "cancelled"
    assert blocked.used == state.used and previous_jobs == set(engine.platform.root.glob("*/request.json"))
    assert weak.evidence in engine.evidence.verify_tree(decision)


def test_negative_boundary_narrows_next_probe(tmp_path, setup_project):
    repo, config = setup_project
    (repo / "evaluate.py").write_text(PROGRAM.replace("params['alpha'] +", "(params['alpha'] if params['alpha'] < 3 else -3) +"))
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qam", "Give the outer setting a negative outcome"], check=True)
    value = config.model_dump()
    value.update(branches=1, max_experiments=4, independent_review=False, stagnation_batches=3)
    value["metrics"][0]["threshold"] = 100
    value["command_experiment"]["choices"]["alpha"] = [0, 1, 2, 3]
    engine = Research.initialize(tmp_path / "narrow", Project.model_validate(value))
    engine.worker = CommandWorker()
    engine.radar = RadarDouble(engine.evidence)
    end = drive(engine, {"complete"})
    plan = engine.evidence.get(end.active_plan)
    assert plan["content"]["details"]["local_range"] == [0, 2]
    assert plan["content"]["details"]["tested_value"] == 2
    assert any(engine.evidence.get(key)["content"].get("metrics") == {"quality": 7} for key in plan["parents"])


def test_no_winner_replay_survives_a_new_controller(tmp_path, setup_project):
    repository, config = setup_project
    # All alternatives measurably regress, so selecting a winner would violate the protocol.
    (repository / "evaluate.py").write_text(
        PROGRAM.replace("+ params['alpha'] + params['beta']", "- params['alpha'] - params['beta']")
    )
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qam", "TEST: every candidate should regress"], check=True)
    value = config.model_dump()
    value.update(max_experiments=3, followup_experiments=False)
    engine = Research.initialize(tmp_path / "decline", Project.model_validate(value))

    class NoWinnerWorker(CommandWorker):
        def run(self, role, payload, response, **kwargs):
            if role == "reviewer":
                self.calls.append(role)
                return IndependentReview(selected_evidence=None, evidence_ids=payload["evidence_ids"],
                                         rationale="TEST confirms no improvement", concerns=[]), {"TEST_DOUBLE": True}
            return super().run(role, payload, response, **kwargs)

    engine.worker = engine.reviewer = NoWinnerWorker()
    engine.radar = RadarDouble(engine.evidence)
    drive(engine, {"converge"})
    state = engine.tick()
    assert state.review_receipt and state.used == 3
    assessment = engine.evidence.get(state.review_check)["content"]
    assert assessment["expected_evidence"] is None and assessment["replay_evidence"]
    restarted = Research(engine.root, worker=engine.worker, radar=engine.radar)
    end = drive(restarted, {"complete"})
    assert end.used == 3
    assert end.best_evidence == end.baseline_evidence
    assert engine.worker.calls.count("reviewer") == 1
    ref = json.loads((engine.root / "reviews/000001-replay.json").read_text())["evidence"]
    assert engine.evidence.get(ref)["content"]["matched"]


def test_escape_research_launches_remaining_dimension(tmp_path, setup_project):
    from loop_research.models import DirectionProposal, LiteratureScreen, PaperQuote, SourceRejection

    repository, config = setup_project
    program = PROGRAM.replace("+ params['alpha'] + params['beta']", "- params['alpha'] - params['beta']")
    program = program.replace("- params['gamma']", "- params['gamma'] + params['delta']")
    (repository / "evaluate.py").write_text(program)
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qam", "TEST: make only the untried dimension improve"], check=True)
    value = config.model_dump()
    value.update(max_experiments=5, followup_experiments=False, independent_review=False, stagnation_batches=1)
    value["metrics"][0]["threshold"] = 11
    value["command_experiment"]["baseline"]["delta"] = 0
    value["command_experiment"]["choices"]["delta"] = [0, 1]
    engine = Research.initialize(tmp_path / "escape", Project.model_validate(value))

    class EscapeWorker(CommandWorker):
        def run(self, role, payload, response, **kwargs):
            if role != "literature-screen":
                return super().run(role, payload, response, **kwargs)
            names = ["alpha", "beta", "gamma"] if payload["mode"] != "escape" else ["delta"]
            directions = [DirectionProposal(
                title=f"TEST change {name}", machine_target=f"parameter.{name}",
                hypothesis="A single configurable variable might change the test metric.",
                rationale="TEST: verify dispatch to the evidence-supported axis.", parameter_bounds="Either zero or one.",
                quotes=[PaperQuote(source_id=paper["source_id"], field="body", quote=paper["body_excerpt"])],
                priority=100-index, limitations=["This is fixture screening."])
                for index, (name, paper) in enumerate(zip(names, payload["papers"]))
            ]
            rejections = [SourceRejection(source_id=paper["source_id"], reason="Unused in this fixture")
                          for paper in payload["papers"][len(directions):]]
            return LiteratureScreen(summary="TEST", directions=directions, excluded_sources=rejections, unresolved=[]), {"TEST_DOUBLE": True}

    engine.worker = engine.reviewer = EscapeWorker()
    engine.radar = RadarDouble(engine.evidence)
    finished = drive(engine, {"complete"})
    assert finished.used == 4 and finished.batch == 2
    assert [task.hypothesis.machine_target for task in finished.active_tasks] == ["parameter.delta"]
    assert finished.best_metrics == {"quality": 11}
    assert finished.offline_accepted


@pytest.mark.parametrize("independent", [False, True])
def test_grouped_study_respects_optional_independent_holdout(tmp_path, setup_project, independent):
    repository, config = setup_project
    (repository / "evaluate.py").write_text(
        """import json
import os
from pathlib import Path
splits = json.loads(Path(os.environ['LOOP_SPLIT_MANIFEST']).read_text())
result = {'metrics': {'quality': 12}}
if os.environ['LOOP_SPLIT'] == 'validation':
    assert list(splits) == ['folds']
    result['fold_results'] = [{'id': fold['id'], 'quality': 12} for fold in splits['folds']]
else:
    assert splits == {'train': ['a', 'b'], 'test': ['c']}
Path(os.environ['LOOP_METRICS']).write_text(json.dumps(result))
"""
    )
    heldout = {"train": ["a", "b"], "test": ["c"]} if independent else None
    (repository / "splits.json").write_text(json.dumps({
        "folds": [{"id": "f1", "train": ["a"], "validation": ["b"]},
                  {"id": "f2", "train": ["b"], "validation": ["a"]}],
        "heldout": heldout,
    }))
    subprocess.run(["git", "-C", str(repository), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qam", "TEST grouped split protocol"], check=True)
    engine = Research.initialize(tmp_path / "grouped", config)
    result = drive(engine, {"complete"})
    assert result.offline_accepted is independent
    assert bool(result.final_receipt) is independent
    if not independent:
        assert result.stop_reason == "independent_holdout_unavailable"
    baseline = engine.evidence.get(result.baseline_evidence)["content"]
    assert [fold["id"] for fold in baseline["fold_results"]] == ["f1", "f2"]
