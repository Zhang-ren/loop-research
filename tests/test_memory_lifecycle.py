"""Explicitly labeled human-action fixtures; never approves live research memory."""
import json

import pytest

from loop_research.memory import Memory
from loop_research.models import MemoryDraft
from loop_research.storage import Evidence, digest
from loop_research.workers import WorkerFailure


class MemoryReviewerDouble:
    def __init__(self, verdict="grounded"):
        self.verdict = verdict
        self.calls = 0

    def run(self, role, payload, response, **kwargs):
        assert role == "memory-logic-review"
        self.calls += 1
        if self.verdict is None:
            raise WorkerFailure("TEST independent worker unavailable")
        return response(verdict=self.verdict, observed_result="TEST: What was observed on this split.",
                        explanation="TEST: Assess the explanation separately.",
                        limitations=["TEST results only"], evidence_ids=payload["evidence_ids"]), {"test_double": True}


@pytest.fixture
def memory(tmp_path):
    evidence = Evidence(tmp_path)
    observations = [evidence.put("tool_observation", {
        "invocation_id": f"TEST-tool-{n}", "verified_by_controller": True,
        "raw_sha256": digest(f"TEST raw log {n}".encode()), "outcome": "TEST recovery verified",
    }) for n in range(2)]
    return Memory(tmp_path, evidence), observations


def draft(observations, content="TEST FIXTURE: reread a task receipt before retrying an infrastructure failure."):
    return MemoryDraft(title="receipt recovery", content=content, evidence_ids=observations,
                       applicability="TEST PROJECT: expired training receipts only", kind="skill")


def approve(memory, key):
    memory.review(key, MemoryReviewerDouble())
    return memory.approve(key, reviewer="TEST HUMAN ONLY", expected_hash=key,
                          explanation="Unit fixture; no actual user approval is asserted")


def test_update_keeps_previous_snapshot_until_review_and_can_rollback(memory):
    m, observations = memory
    key = m.propose(draft(observations), "project-A", 1)
    assert m.screen(key)["passed"]
    assert m.search("receipt", "project-A") == []
    approve(m, key)
    old_path = m.root / "skills" / m.name(key) / "SKILL.md"
    old_bytes = old_path.read_bytes()
    update = m.propose(draft(observations, "TEST FIXTURE: reread durable receipts, reconcile unknown submissions, then retry."),
                       "project-A", 2, supersedes=key)
    assert [r["id"] for r in m.search("receipt", "project-A")] == [key]
    assert old_path.read_bytes() == old_bytes
    approve(m, update)
    assert not old_path.exists()
    assert [r["id"] for r in m.search("receipt", "project-A")] == [update]
    third = m.propose(draft(observations, "TEST FIXTURE: classify submission outcome, reread receipts and reconcile before retrying."),
                      "project-A", 3, supersedes=update)
    approve(m, third)
    assert [r["id"] for r in m.search("receipt", "project-A")] == [third]
    m.rollback(key, reviewer="TEST HUMAN ONLY", explanation="Fixture rollback")
    assert old_path.read_bytes() == old_bytes
    assert [r["id"] for r in m.search("receipt", "project-A")] == [key]
    m.revoke(key, "TEST counterexample")
    assert not old_path.exists() and not m.search("receipt", "project-A")
    with pytest.raises(ValueError, match="Revoked"):
        m.rollback(key, reviewer="TEST HUMAN ONLY", explanation="Must not resurrect")


def test_revision_worker_can_draft_only_with_same_scope_and_supplied_evidence(memory):
    m, observations = memory
    key = m.propose(draft(observations), "project-A", 1)
    approve(m, key)
    approved = m.root / "skills" / m.name(key) / "SKILL.md"
    original = approved.read_bytes()

    class RevisionWorker:
        def run(self, role, payload, response, **kwargs):
            assert role == "memory-reviser" and payload["evidence_ids"] == observations
            return draft(observations, "TEST FIXTURE: reconcile the durable receipt before retrying the same task."), {}

    result = m.revise(key, "Clarify when the original advice applies.", [], 2, RevisionWorker())
    candidate = m.read(result["proposal"])
    assert candidate["supersedes"] == key
    assert m.evidence.get(result["revision_evidence"])["content"]["proposal"] == result["proposal"]
    assert approved.read_bytes() == original
    assert [record["id"] for record in m.search("receipt", "project-A")] == [key]


def test_curation_preserves_small_class_counts_without_exposing_predictions():
    from loop_research.memory import curation_observation

    def render(predictions):
        return curation_observation("measure", {"kind": "training_result",
                                    "content": {"predictions": predictions}})["content"]

    compact = render([3] * 359)
    assert compact["prediction_label_counts"] == [{"label": 3, "count": 359}]
    assert compact["arrays"]["predictions"]["count"] == 359 and "predictions" not in compact
    assert "prediction_label_counts" not in render(list(range(33)))
    assert "prediction_label_counts" not in render([0.25, 0.5])


def test_fabricated_summary_cannot_pass_screening_or_leak_via_index(memory):
    m, _ = memory
    claim = m.evidence.put("agent_summary", {"metrics": {"accuracy": 1.0}, "validated": True})
    key = m.propose(draft([claim]), "project-A", 1)
    assert not m.screen(key)["passed"]
    with pytest.raises(ValueError, match="External validation"):
        approve(m, key)
    assert not list((m.root / "skills").glob("*/SKILL.md"))
    assert not m.search("receipt", "project-A")
    assert m.timeline("project-A")[0]["trust"] == "candidate_unverified"


def test_upstream_format_screen_runs_before_human_publication(memory):
    m, observations = memory
    key = m.propose(draft(observations, "TEST FIXTURE: TODO fill missing recovery procedure before any publication."), "project-A", 1)
    result = m.screen(key)
    assert not result["passed"] and any("TODO" in message for message in result["errors"])
    assert not list((m.root / "skills").glob("*/SKILL.md"))


def test_valid_observations_cannot_publish_an_unsupported_explanation(memory):
    m, observations = memory
    key = m.propose(draft(observations, "TEST FIXTURE: always attribute budget savings to this recovery rule."),
                    "project-A", 1)
    assert m.screen(key)["passed"]
    with pytest.raises(ValueError, match="independent memory review"):
        m.approve(key, reviewer="TEST HUMAN ONLY", expected_hash=key, explanation="Test missing logic review")
    unavailable = m.review(key, MemoryReviewerDouble(None))
    assert unavailable["status"] == "reviewer_unavailable"
    review = m.review(key, MemoryReviewerDouble("unsupported"))
    assert review["status"] == "awaiting_evidence"
    restarted = Memory(m.root.parent, m.evidence)
    unused = MemoryReviewerDouble()
    assert restarted.logic_review(key)["assessment"]["verdict"] == "unsupported"
    assert restarted.review(key, unused) == review
    assert unused.calls == 0
    with pytest.raises(ValueError, match="independent memory review"):
        restarted.approve(key, reviewer="TEST HUMAN ONLY", expected_hash=key, explanation="Test rejected logic")
    assert not list((m.root / "skills").glob("*/SKILL.md"))
    assert restarted.search("receipt", "project-A") == []


def test_approved_files_and_evidence_are_checked_on_every_retrieval(memory):
    m, observations = memory
    key = m.propose(draft(observations), "project-A", 1)
    approve(m, key)
    path = m.root / "skills" / m.name(key) / "SKILL.md"
    path.write_text(path.read_text() + "unreviewed edit")
    with pytest.raises(ValueError, match="changed without review"):
        m.search("receipt", "project-A")


def test_cross_project_update_rejected_and_index_rebuilt_from_files(memory):
    m, observations = memory
    key = m.propose(draft(observations), "project-A", 1)
    approve(m, key)
    with pytest.raises(ValueError, match="scope"):
        m.propose(draft(observations), "project-B", 2, supersedes=key)
    (m.root / "index.sqlite").unlink()
    assert m.search("receipt", "project-A")
    assert not m.search("receipt", "project-B")
    candidate = m.root / "candidates" / f"{key}.json"
    payload = json.loads(candidate.read_bytes())
    payload["draft"]["content"] = "unreviewed candidate change"
    candidate.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="changed since submission"):
        m.search("receipt", "project-A")
