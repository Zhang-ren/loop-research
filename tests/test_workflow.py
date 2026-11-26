"""Controller tests with labeled worker doubles and real detached training.

These do not establish real Codex research quality; the live demo is separate.
"""
import json
import time
from pathlib import Path

import pytest

from loop_research.engine import Research
from loop_research.models import (Analysis, CodeResult, Curation, DirectionProposal, Hypothesis, OrthogonalityReview,
                                 LiteratureScreen, PaperQuote, Plan, Project, ResearchQuery, ResearchRequest)
from loop_research.storage import Conflict
from loop_research.training import BASELINE_CODE


class WorkerDouble:
    def __init__(self):
        self.calls = []

    def run(self, role, payload, response, **kwargs):
        self.calls.append(role)
        metadata = {"TEST_DOUBLE": True, "role": role}
        if role == "research-query":
            number = len(payload["previous_queries"])
            return ResearchRequest(rationale="TEST FIXTURE research question", criteria=["TEST method", "TEST applicability"],
                queries=[ResearchQuery(query=f"TEST query {number} {letter}", gap="TEST missing method knowledge") for letter in "AB"]), metadata
        if role == "literature-screen":
            targets = [t for t in ["preprocessing", "classifier.LogisticRegression.C", "classifier.family"]
                       if t in payload["allowed_machine_targets"]]
            if len(targets) < 3:
                targets = payload["allowed_machine_targets"][:3]
            directions = [DirectionProposal(title="TEST supported direction " + target, machine_target=target,
                hypothesis="TEST FIXTURE: check measured improvement under the frozen split", rationale="TEST FIXTURE paper-backed proposal",
                parameter_bounds="TEST bounded intervention", quotes=[PaperQuote(source_id=paper["source_id"], field="body", quote=paper["body_excerpt"])],
                priority=90-i, limitations=["TEST sources are not actual literature"],
                classifier_name="RandomForestClassifier" if target == "classifier.family" else None)
                for i, (target, paper) in enumerate(zip(targets, payload["papers"], strict=True))]
            return LiteratureScreen(summary="TEST screening", directions=directions, excluded_sources=[], unresolved=[]), metadata
        if role == "curator":
            return Curation(summary="TEST FIXTURE; no live publication", drafts=[], unresolved=["human review"]), metadata
        if role == "orthogonality-review":
            return OrthogonalityReview(overlaps=[]), metadata
        if role == "planner":
            labels = {"preprocessing": "scaling", "classifier.LogisticRegression.C": "regularization", "classifier.family": "classifier_family"}
            hypotheses = [Hypothesis(title=f"direction-{i}", question="Does this change help?", axis=labels[d["machine_target"]],
                          change=labels[d["machine_target"]], prediction="potential improvement", falsification="no gain under fixed split",
                          source_ids=payload["source_ids"], parameter_bounds="bounded by fixed protocol", single_variable=True,
                          machine_target=d["machine_target"], direction_id=d["id"], classifier_name=d.get("classifier_name"))
                          for i, d in enumerate(payload["direction_pool"][:payload["branches"]])]
            return Plan(rationale="TEST FIXTURE", hypotheses=hypotheses), metadata
        if role == "coder":
            hypothesis = payload["hypothesis"]
            axis = hypothesis["axis"]
            if axis == "scaling":
                code = BASELINE_CODE.replace("from sklearn.pipeline", "from sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline").replace("make_pipeline(LogisticRegression", "make_pipeline(StandardScaler(), LogisticRegression")
            elif axis == "regularization":
                code = BASELINE_CODE.replace("C=1.0", "C=0.1")
            else:
                code = "from sklearn.ensemble import RandomForestClassifier\ndef build_model(seed):\n    return RandomForestClassifier(n_estimators=40, random_state=seed)\n"
            return CodeResult(rationale="TEST FIXTURE", code=code, source_ids=hypothesis["source_ids"], changed_axis=axis), metadata
        return Analysis(summary="TEST FIXTURE: compare the supplied measured results", evidence_ids=payload["evidence_ids"],
                        explanation="No causal proof from a single split", confidence=0.5, next_directions=["replicate"],
                        limitations=["worker double"], metric_tradeoffs="all thresholds independently checked"), metadata


class RadarDouble:
    """Labeled data-only fixture; real paper retrieval is verified separately."""
    def __init__(self, evidence):
        self.evidence, self.queries = evidence, []

    def history(self):
        return self.queries

    def search(self, queries):
        self.queries.extend(queries)
        search = self.evidence.put("paper_search", {"TEST_DOUBLE": True, "queries": queries})
        sources = [self.evidence.put("paper_source", {"title": f"TEST source {i}", "retracted": False,
            "access": "body_excerpt", "abstract": "TEST FIXTURE externally supplied abstract.",
            "body_excerpt": f"TEST FIXTURE paper method evidence {i}, not actual scholarly literature."}, [search]) for i in range(3)]
        return {"searches": [{"evidence": search}], "source_ids": sources}

    def cards(self, source_ids):
        return [{"source_id": key, **self.evidence.get(key)["content"]} for key in source_ids]


def prepare_research(engine):
    engine.worker = WorkerDouble()
    engine.radar = RadarDouble(engine.evidence)
    for _ in range(4):
        engine.tick()
    assert engine.store.read().phase == "plan" and engine.store.read().direction_pool
    engine.worker.calls.clear()
    return engine


@pytest.fixture
def engine(tmp_path):
    project = Project.model_validate_json((Path(__file__).parents[1] / "examples/breast-cancer.json").read_bytes())
    project.max_experiments = 3
    engine = Research.initialize(tmp_path, project)
    return prepare_research(engine)


def test_spawn_async_restart_converge_and_budget(engine):
    state = engine.tick(event_id="start")
    assert state.phase == "execute" and len(state.active_tasks) == 3
    revision = state.revision
    assert engine.tick(event_id="start").revision == revision
    state = engine.tick()
    assert state.phase == "monitor" and all(t.receipt for t in state.active_tasks)
    # A new controller instance has no previous conversation or in-memory graph state.
    new = Research(engine.root, worker=engine.worker)
    deadline = time.time() + 40
    while new.store.read().phase == "monitor" and time.time() < deadline:
        new.tick()
        time.sleep(0.1)
    assert new.store.read().phase == "converge"
    assert new.store.read().used == 3
    state = new.tick()
    assert state.phase == "curate"
    assert new.tick().phase == "finalize"
    deadline = time.time() + 40
    while new.store.read().phase == "finalize" and time.time() < deadline:
        new.tick()
        time.sleep(0.1)
    state = new.store.read()
    assert state.phase == "complete" and state.final_evidence
    holdout = new.evidence.get(state.final_evidence)["content"]
    selected = new.evidence.get(state.best_evidence)["content"]
    assert not set(holdout["evaluation_indices"]) & set(selected["evaluation_indices"])
    assert state.offline_accepted == all(m.passes(holdout["metrics"][m.name]) for m in engine.project.metrics)
    assert state.used == 3
    before = list(engine.worker.calls)
    new.tick()
    assert engine.worker.calls == before
    assert new.report().exists()
    assert len(list((engine.root / "memory/candidates").glob("*.json"))) == 1
    assert new.memory.search("findings", state.project_id) == []
    for task in state.active_tasks:
        assert task.code_commit and task.evidence
        new.evidence.verify_tree(task.evidence)


def test_active_lease_blocks_duplicate_decisions(engine):
    engine.store.change(lambda s: (setattr(s, "lease", "other"), setattr(s, "lease_until", time.time()+100)))
    with pytest.raises(Conflict):
        engine.tick()
    assert engine.worker.calls == []


def test_event_retains_a_commit_after_restart_and_retries_precommit_failure(engine, monkeypatch):
    original = engine.graph.invoke

    def fail_before_commit(_):
        raise RuntimeError("TEST injected wake failure")

    monkeypatch.setattr(engine.graph, "invoke", fail_before_commit)
    with pytest.raises(RuntimeError, match="TEST injected"):
        engine.tick(event_id="upstream-result-1")
    assert not engine.store.has_event("upstream-result-1", engine.store.read())
    monkeypatch.setattr(engine.graph, "invoke", original)
    assert engine.tick(event_id="upstream-result-1").phase == "execute"
    engine.store.change(lambda s: setattr(s, "event_ids", []), event="TEST event left the recent cache")
    restarted = Research(engine.root, worker=engine.worker)
    revision = restarted.store.read().revision
    assert restarted.tick(event_id="upstream-result-1").revision == revision
    assert restarted.store.has_event("upstream-result-1", restarted.store.read())


def test_overlapping_semantics_block_distinct_labeled_branches(engine):
    class SemanticOverlap(WorkerDouble):
        def run(self, role, payload, response, **kwargs):
            if role == "orthogonality-review":
                targets = [row["machine_target"] for row in payload["hypotheses"][:2]]
                assert len(set(targets)) == 2
                return response(overlaps=[{"machine_targets": targets,
                    "reason": "TEST independent assessment identifies the same intervention."}]), {"TEST_DOUBLE": True}
            return super().run(role, payload, response, **kwargs)

    engine.worker = SemanticOverlap()
    state = engine.tick()
    assert state.phase == "plan" and state.batch == 0 and state.active_tasks == []
    assert not list((engine.root / "jobs").glob("*/request.json"))
    review = engine.evidence.get(state.last_rejection)
    assert review["kind"] == "plan_orthogonality_review" and not review["content"]["accepted"]


def test_protocol_mutation_rejected(engine):
    value = json.loads((engine.root / "project.json").read_bytes())
    value["metrics"][0]["threshold"] = 0.01
    (engine.root / "project.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frozen"):
        Research(engine.root).tick()


def test_exhausted_branch_does_not_hide_completed_sibling_measurements(engine):
    engine.tick()
    engine.tick()
    tasks = engine.store.read().active_tasks
    deadline = time.time() + 40
    while time.time() < deadline:
        if all(engine.platform.poll(t.receipt)["status"] == "succeeded" for t in tasks):
            break
        time.sleep(0.1)
    def inject_failure(state):
        state.active_tasks[0].status = "infra_failed"
        state.active_tasks[0].attempts = engine.project.infrastructure_retries + 1
        state.active_tasks[0].error = "TEST FIXTURE: exhausted infrastructure branch"
    engine.store.change(inject_failure, event="TEST injected failure")
    state = engine.tick()
    assert state.phase == "review"
    assert state.used == 2
    assert all(t.status == "succeeded" and t.counted and t.evidence for t in state.active_tasks[1:])


def test_correct_axis_label_cannot_authorize_a_different_actual_parameter(engine):
    original = engine.worker
    class WrongIntervention:
        def run(self, role, payload, response, **kwargs):
            result, meta = original.run(role, payload, response, **kwargs)
            if role == "coder" and payload["hypothesis"]["axis"] == "regularization":
                result.code = BASELINE_CODE.replace("random_state=seed", "random_state=seed, class_weight='balanced'")
            return result, meta
    engine.worker = WrongIntervention()
    engine.tick()
    state = engine.tick()
    rejected = next(t for t in state.active_tasks if t.hypothesis.axis == "regularization")
    assert rejected.status == "contract_rejected" and not rejected.receipt
    assert "machine_target" in rejected.error


def test_training_failure_disposition_does_not_exempt_ambiguous_failures():
    import errno
    import subprocess

    from loop_research.runtime_failure import classify_failure

    assert classify_failure(ValueError("invalid model parameter"))[0] == "code_failed"
    assert classify_failure(subprocess.TimeoutExpired(["train"], 1))[0] == "code_failed"
    assert classify_failure(OSError(errno.ENOSPC, "no space"))[0] == "infra_failed"
    assert classify_failure(FileNotFoundError(errno.ENOENT, "unknown missing input"))[0] == "unclassified_failed"
    assert classify_failure(subprocess.CalledProcessError(-9, ["train"]))[0] == "unclassified_failed"


def test_overlapping_actual_parameters_replan_before_any_training(engine):
    original = engine.worker
    class Overlapping:
        def run(self, role, payload, response, **kwargs):
            result, meta = original.run(role, payload, response, **kwargs)
            if role == "planner":
                result.hypotheses[2].machine_target = "classifier.LogisticRegression.C"
                result.hypotheses[2].direction_id = result.hypotheses[1].direction_id
                result.hypotheses[2].classifier_name = None
            elif role == "coder" and payload["hypothesis"]["axis"] == "classifier_family":
                result.code = BASELINE_CODE.replace("C=1.0", "C=10.0")
            return result, meta
    engine.worker = Overlapping()
    engine.tick()
    state = engine.tick()
    assert state.phase == "plan" and state.last_rejection
    assert not list(engine.platform.root.glob("*/request.json"))
    assert state.used == 0


def test_paper_supported_classifier_cannot_be_silently_replaced(engine):
    original = engine.worker
    class WrongFamily:
        def run(self, role, payload, response, **kwargs):
            result,meta = original.run(role,payload,response,**kwargs)
            if role == 'coder' and payload['hypothesis']['machine_target'] == 'classifier.family':
                result.code = 'from sklearn.svm import SVC\ndef build_model(seed):\n    return SVC(random_state=seed)\n'
            return result,meta
    engine.worker = WrongFamily()
    engine.tick()
    state = engine.tick()
    task = next(t for t in state.active_tasks if t.hypothesis.machine_target == 'classifier.family')
    assert task.hypothesis.classifier_name == 'RandomForestClassifier'
    assert task.status == 'contract_rejected' and not task.receipt
    assert 'paper-supported direction' in task.error


def test_analysis_cannot_omit_a_completed_branch_and_receives_all_three_evidence_layers(engine):
    engine.tick()
    engine.tick()
    deadline = time.time()+40
    while engine.store.read().phase == 'monitor' and time.time()<deadline:
        engine.tick()
        time.sleep(0.1)
    assert engine.store.read().phase == 'converge'
    original = engine.worker
    class OmittedBranch:
        def run(self, role, payload, response, **kwargs):
            result,meta = original.run(role,payload,response,**kwargs)
            if role == 'analyst':
                assert payload['confirmed_knowledge']['evidence_id'] in payload['evidence_ids']
                assert len(payload['screened_literature_context']) == 3
                assert all(d['quotes'] for d in payload['screened_literature_context'])
                omitted = payload['required_measurement_citations'][-1]
                result.evidence_ids = [key for key in result.evidence_ids if key != omitted]
            return result,meta
    engine.worker = OmittedBranch()
    engine.tick()
    state = engine.store.read()
    assert state.phase == 'converge' and not state.last_analysis and state.analysis_failures == 1
    assert 'omitted a completed branch' in engine.evidence.get(state.last_analysis_rejection)['content']['reason']
    engine.worker = original
    state = engine.tick()
    assert state.phase == 'curate' and state.analysis_failures == 0


def test_repeated_invalid_analysis_stops_after_bounded_retries_without_new_training(engine):
    engine.tick()
    engine.tick()
    deadline=time.time()+40
    while engine.store.read().phase=='monitor' and time.time()<deadline:
        engine.tick()
        time.sleep(0.1)
    original=engine.worker
    class InvalidCitation:
        def run(self,role,payload,response,**kwargs):
            result,meta=original.run(role,payload,response,**kwargs)
            if role=='analyst':
                result.evidence_ids=['INVALID-TEST-REFERENCE']
            return result,meta
    engine.worker=InvalidCitation()
    for _ in range(3):
        state=engine.tick()
    assert state.phase=='review' and state.analysis_failures==3
    assert state.stop_reason=='analysis_contract_retry_limit' and not state.last_analysis
    assert state.used==3 and len(list(engine.platform.root.glob('*/request.json')))==3


def test_curator_loads_current_batch_and_only_five_prior_failures(engine):
    from loop_research.storage import atomic,canonical
    engine.tick()
    engine.tick()
    deadline=time.time()+40
    while engine.store.read().phase=='monitor' and time.time()<deadline:
        engine.tick()
        time.sleep(0.1)
    engine.tick()
    state=engine.store.read()
    assert state.phase=='curate'
    def metadata(name,started,exit_code):
        folder=engine.root/'workers'/name
        atomic(folder/'metadata.json',canonical({'call_id':name,'role':'coder','started':started,'exit_code':exit_code,'reason':None}))
        atomic(folder/'events.jsonl',b'TEST FIXTURE raw log')
        atomic(folder/'stderr.log',b'')
    for i in range(7):
        metadata(f'TEST-prior-failure-{i}',state.batch_started-100+i,1)
    for i in range(15):
        metadata(f'TEST-old-success-{i}',state.batch_started-50+i,0)
    for i in range(2):
        metadata(f'TEST-current-{i}',state.batch_started+1+i,0)
    original=engine.worker
    class Capture:
        def run(self,role,payload,response,**kwargs):
            if role=='curator':
                ids={o['content']['invocation_id'] for o in payload['observations'] if o['kind']=='tool_observation'}
                assert ids == {f'TEST-prior-failure-{i}' for i in range(2,7)} | {f'TEST-current-{i}' for i in range(2)}
            return original.run(role,payload,response,**kwargs)
    engine.worker=Capture()
    assert engine.tick().phase=='finalize'
