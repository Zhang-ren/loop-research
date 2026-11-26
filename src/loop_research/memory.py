from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from pathlib import Path

from filelock import FileLock
from pydantic import ValidationError

from ._vendor.evo import proposals
from .models import MemoryDraft, MemoryLogicAssessment
from .storage import Evidence, StateStore, atomic, canonical, digest
from .workers import WorkerFailure


MEASUREMENT_KINDS = {"training_result", "measured_baseline", "holdout_result"}
CURRENT_TOOL_OBSERVATIONS = 24
PRIOR_TOOL_FAILURES = 5


def curation_tool_records(entries, started: float):
    """Select references to nearby tool activity while leaving the full journal intact."""
    ordered = sorted(entries, key=lambda item: item[1]["started"])
    recent = [(path, meta) for path, meta in ordered if meta["started"] >= started]
    failures = [(path, meta) for path, meta in ordered
                if meta["started"] < started and (meta["exit_code"] != 0 or meta.get("reason"))]
    return failures[-PRIOR_TOOL_FAILURES:] + recent[-CURRENT_TOOL_OBSERVATIONS:]


def prediction_label_counts(predictions):
    if not isinstance(predictions, list) or not predictions:
        return None
    counts = {}
    for label in predictions:
        if type(label) not in (int, str) or (isinstance(label, str) and len(label) > 64):
            return None
        counts[label] = counts.get(label, 0) + 1
        if len(counts) > 32:
            return None
    return [{"label": label, "count": count} for label, count in counts.items()]


def curation_observation(key: str, evidence: dict) -> dict:
    if evidence["kind"] not in MEASUREMENT_KINDS:
        return {"id": key, **evidence}
    measured = evidence["content"]
    content = {name: measured[name] for name in (
        "status", "metrics", "seed", "split", "code_sha256", "data_sha256", "source_revision", "logs", "failure_category",
    ) if name in measured}
    if isinstance(measured.get("error"), str):
        content["error"] = measured["error"][-2000:]
    content["arrays"] = {name: {"count": len(measured[name]), "sha256": digest(canonical(measured[name]))}
                         for name in ("train_indices", "evaluation_indices", "predictions", "scores", "fold_results")
                         if isinstance(measured.get(name), list)}
    if counts := prediction_label_counts(measured.get("predictions")):
        content["prediction_label_counts"] = counts
    return {"id": key, **evidence, "content": content}


class Memory:
    """Files are authoritative, FTS disposable. Evo handles skill format/publication.

    Domain-specific scope, evidence and human-review gates live outside the Agent.
    Neither a model summary nor a caller's boolean certifies an experience.
    """
    def __init__(self, root: Path, evidence: Evidence, *, project_scope: str | None = None):
        self.root = root / "memory"
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence = evidence
        self.lock = FileLock(str(self.root / ".lock"))
        self.project_scope = project_scope

    def require_scope(self, project):
        if self.project_scope is not None and project != self.project_scope:
            raise ValueError("Memory belongs to a different research project")

    @staticmethod
    def name(key):
        return "research-" + key[:48]

    def event(self, key, action, **details):
        value = {"proposal": key, "action": action, "time": time.time(), **details}
        atomic(self.root / "events" / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.json", canonical(value))

    def propose(self, draft: MemoryDraft, project: str, batch: int, *, supersedes: str | None = None) -> str:
        self.require_scope(project)
        for key in draft.evidence_ids:
            self.evidence.verify_tree(key)
        if supersedes:
            old = self.read(supersedes, "approved")
            if old["project"] != project or old["draft"]["kind"] != draft.kind:
                raise ValueError("Updates must retain project scope and memory kind")
        record = {"schema": 3, "project": project, "batch": batch,
                  "draft": draft.model_dump(), "supersedes": supersedes}
        key = digest(canonical(record))
        with self.lock:
            legacy_key = digest(canonical({**record, "schema": 2}))
            if (self.root / "candidates" / f"{legacy_key}.json").is_file():
                self.read(legacy_key)
                return legacy_key
            path = self.root / "candidates" / f"{key}.json"
            if not path.exists():
                atomic(path, canonical(record))
                self.event(key, "proposed")
        return key

    def revise(self, previous: str, instruction: str, evidence_ids: list[str], batch: int, worker) -> dict:
        if not instruction.strip():
            raise ValueError("A revision needs a reason or correction")
        active = self.read(previous, "approved")
        if active["status"] != "approved":
            raise ValueError("Only the active approved version can be revised")
        original = active["draft"]
        sources = list(dict.fromkeys([*original["evidence_ids"], *evidence_ids]))
        for key in sources:
            self.evidence.verify_tree(key)
        payload = {
            "previous_proposal": previous,
            "original_draft": original,
            "requested_revision": instruction,
            "evidence_ids": sources,
            "observations": [self._review_observation(key) for key in sources],
            "requirements": (
                "Propose the next version using only the scoped evidence in this project. Preserve the original "
                "memory kind. Retain all still-applicable conditions. Clearly distinguish observed facts from "
                "procedures and hypotheses. Cite the supplied evidence IDs supporting the new draft. "
                "Return a candidate for independent review and human approval."
            ),
        }
        replacement, invocation = worker.run("memory-reviser", payload, MemoryDraft)
        if replacement.kind != original["kind"] or not set(replacement.evidence_ids) <= set(sources):
            raise ValueError("Revision changed the memory kind or cited unprovided evidence")
        if replacement.model_dump() == original:
            raise ValueError("Revision is identical to the active version")
        proposal = self.propose(replacement, active["project"], batch, supersedes=previous)
        trace = self.evidence.put("memory_revision", {"previous": previous, "proposal": proposal,
            "requested_revision": instruction, "worker": invocation}, replacement.evidence_ids)
        return {"proposal": proposal, "supersedes": previous, "revision_evidence": trace}

    def read(self, key: str, area="candidates") -> dict:
        if len(key) != 64 or not all(c in "0123456789abcdef" for c in key):
            raise ValueError("Invalid memory ID")
        raw = (self.root / "candidates" / f"{key}.json").read_bytes()
        if digest(raw) != key:
            raise ValueError("Memory candidate changed since submission")
        proposal = json.loads(raw)
        self.require_scope(proposal["project"])
        if area == "candidates":
            return proposal
        if area != "approved":
            raise ValueError("Invalid memory area")
        decision = json.loads((self.root / "approved" / f"{key}.json").read_bytes())
        if decision["proposal"] != key or decision["project"] != proposal["project"]:
            raise ValueError("Approval does not bind this proposal and project")
        if decision.get("logic_review") or proposal.get("schema", 2) >= 3:
            self.require_grounded(key, decision["screen"], decision.get("logic_review"))
        if decision["status"] == "approved":
            screen = self.evidence.get(decision["screen"])
            if screen["kind"] != "memory_screen" or screen["content"]["proposal"] != key or not screen["content"]["passed"]:
                raise ValueError("Missing evidence screening")
            for evidence_id in proposal["draft"]["evidence_ids"]:
                self.evidence.verify_tree(evidence_id)
            if proposal["draft"]["kind"] == "skill":
                path = self.root / "skills" / self.name(key) / "SKILL.md"
                if not path.exists() or digest(path.read_bytes()) != decision["skill_sha256"]:
                    raise ValueError("Published skill changed without review")
        return {**proposal, **decision, "id": key}

    def screen(self, key: str) -> dict:
        with self.lock:
            return self._screen(key)

    def _screen(self, key: str) -> dict:
        """Check provenance and scope before independent and human review."""
        record = self.read(key)
        draft = MemoryDraft.model_validate(record["draft"])
        verified, errors = set(), []
        for source in draft.evidence_ids:
            for item in self.evidence.verify_tree(source):
                value = self.evidence.get(item)
                content = value["content"]
                if value["kind"] in MEASUREMENT_KINDS:
                    metrics = content.get("metrics", {})
                    if (content.get("status", "succeeded") == "succeeded" and metrics
                            and content.get("code_sha256") and content.get("data_sha256")
                            and all(type(v) in (int, float) and math.isfinite(v) for v in metrics.values())):
                        verified.add(item)
                elif value["kind"] == "tool_observation" and content.get("verified_by_controller") is True:
                    if content.get("invocation_id") and content.get("raw_sha256") and content.get("outcome"):
                        verified.add(item)
        if not verified:
            errors.append("No externally measured experiment or verified tool observation")
        if draft.kind == "skill" and len(verified) < 2:
            errors.append("Reusable skills require at least two distinct external observations")
        if draft.kind == "skill":
            result = self.stage_skill(key, record)
            if not result["submitted"]:
                errors.extend(result.get("errors", [result.get("error", "Upstream format validation failed")]))
        result = {"proposal": key, "passed": not errors, "errors": errors,
                  "external_observations": sorted(verified), "human_review_required": True,
                  "scope": record["project"],
                  "limitation": "Provenance screening cannot certify causal explanations or general applicability."}
        artifact = self.evidence.put("memory_screen", result, draft.evidence_ids)
        atomic(self.root / "screens" / f"{key}.json", canonical({"evidence": artifact}))
        return {**result, "evidence": artifact}

    def logic_review(self, key: str, *, receipt: str | None = None) -> dict | None:
        proposal = self.read(key)
        if receipt is None:
            path = self.root / "logic-reviews" / f"{key}.json"
            if not path.is_file():
                return None
            receipt = json.loads(path.read_bytes())["evidence"]
        record = self.evidence.get(receipt)
        content = record["content"]
        if (record["kind"] != "memory_logic_review" or content["proposal"] != key
                or content["project"] != proposal["project"] or content["screen"] not in record["parents"]):
            raise ValueError("Independent review does not bind this proposal and project")
        return {**content, "evidence": receipt}

    def require_grounded(self, key: str, screen: str, receipt: str | None):
        review = self.logic_review(key, receipt=receipt) if receipt else None
        if not review or review["screen"] != screen or review["status"] != "eligible_for_human_review":
            raise ValueError("A grounded independent memory review is required before promotion")
        if MemoryLogicAssessment.model_validate(review["assessment"]).verdict != "grounded":
            raise ValueError("Independent memory review did not support the proposed claim")
        self.evidence.verify_tree(receipt)

    def review(self, key: str, reviewer, *, context_ids=(), cancel_check=None) -> dict:
        record = self.read(key)
        with FileLock(str(self.root / f".logic-{key}.lock")):
            screen = self.screen(key)
            existing = self.logic_review(key)
            if not screen["passed"]:
                if existing and existing["screen"] == screen["evidence"] and existing["status"] == "screen_rejected":
                    return existing
                return self._record_review(key, record, screen, "screen_rejected")
            draft = MemoryDraft.model_validate(record["draft"])
            controls = []
            for source in screen["external_observations"]:
                value = self.evidence.get(source)
                if value["kind"] in MEASUREMENT_KINDS:
                    controls.extend(parent for parent in value["parents"]
                                    if self.evidence.get(parent)["kind"] == "model_code")
            known = list(dict.fromkeys([*draft.evidence_ids, *screen["external_observations"], *controls, *context_ids]))
            observations = [self._review_observation(source) for source in known]
            payload = {
                "draft": draft.model_dump(),
                "evidence_ids": known,
                "observations": observations,
                "requirements": (
                    "Independently review the exact proposed text and applicability, without editing it. "
                    "State only what the actual observations show in observed_result. In explanation, assess "
                    "support for the proposed attribution, procedure or mechanism separately. Cite provided raw "
                    "measurements or tool observations as appropriate. Identify scope, comparison controls, missing "
                    "replications and plausible alternatives. A shared split or metric gain does not establish a "
                    "general mechanism or cross-project result. Literature and business facts are supporting context. "
                    "Use grounded for a properly scoped supported claim, unsupported for contradicted claims or stronger "
                    "claims the design cannot justify, and inconclusive when relevant evidence is missing or indeterminate."
                ),
            }
            input_sha256 = digest(canonical(payload))
            if existing and existing["screen"] == screen["evidence"] and existing["status"] != "reviewer_unavailable":
                previous_input = existing.get("review_input_sha256")
                if previous_input == input_sha256 or (
                    previous_input is None and existing["status"] == "eligible_for_human_review"
                ):
                    return existing
            worker = None
            try:
                result, worker = reviewer.run("memory-logic-review", payload, MemoryLogicAssessment,
                                              cancel_check=cancel_check)
                cited = set(result.evidence_ids)
                required = 2 if draft.kind == "skill" and result.verdict == "grounded" else 1
                if not cited <= set(known) or len(cited.intersection(screen["external_observations"])) < required:
                    raise WorkerFailure("Independent memory review did not cite the required verified observations")
            except (WorkerFailure, OSError, ValidationError) as error:
                failure = "Independent reviewer returned invalid structured output" if isinstance(error, ValidationError) else str(error)
                return self._record_review(key, record, screen, "reviewer_unavailable", sources=known,
                                           failure=failure, worker=worker, review_input_sha256=input_sha256)
            status = "eligible_for_human_review" if result.verdict == "grounded" else "awaiting_evidence"
            human_action = ("review_exact_proposal" if result.verdict == "grounded"
                            else "review_limitations_and_supply_evidence_or_revise_claim")
            return self._record_review(key, record, screen, status, sources=known,
                                       assessment=result.model_dump(), worker=worker, human_action=human_action,
                                       review_input_sha256=input_sha256)

    def _record_review(self, key, record, screen, status, *, sources=(), **details):
        result = {"proposal": key, "project": record["project"], "screen": screen["evidence"],
                  "status": status, **details}
        artifact = self.evidence.put("memory_logic_review", result, list(dict.fromkeys([screen["evidence"], *sources])))
        atomic(self.root / "logic-reviews" / f"{key}.json", canonical({"evidence": artifact}))
        self.event(key, "logic_review", status=status, evidence=artifact)
        return {**result, "evidence": artifact}

    def _review_observation(self, key):
        source = self.evidence.get(key)
        kind = source["kind"]
        if kind in MEASUREMENT_KINDS:
            return curation_observation(key, source)
        if kind == "model_code":
            code = source["content"].get("code")
            if isinstance(code, str):
                return {"id": key, "kind": kind, "parents": source["parents"],
                        "content": {"code_excerpt": code[:3000], "sha256": digest(code.encode()),
                                    "excerpt_truncated": len(code) > 3000}}
        fields = {
            "confirmed_business_knowledge": ("text",),
            "paper_source": ("title", "url", "access", "retracted", "abstract", "body_excerpt"),
            "tool_observation": ("invocation_id", "outcome", "role", "raw_sha256", "error_excerpt",
                                 "verified_by_controller"),
            "native_repository_control": ("parameter", "evidence_path", "rationale"),
        }.get(kind, ())
        content = {field: value[:6000] if isinstance(value, str) else value
                   for field in fields if (value := source["content"].get(field)) is not None}
        if kind == "tool_observation" and content.get("verified_by_controller") is True:
            code = source["content"].get("metadata", {}).get("exit_code")
            if type(code) is int:
                content["exit_code"] = code
        return {"id": key, "kind": kind, "parents": source["parents"], "content": content}

    def stage_skill(self, key, record):
        name = self.name(key)
        directory = self.root / "autoskills" / "proposals" / name
        text = ("---\nname: " + name + "\ndescription: " + json.dumps(record["draft"]["title"], ensure_ascii=False)
                + "\n---\n\nApplicability: " + record["draft"]["applicability"] + "\n\n"
                + record["draft"]["content"] + "\n\nEvidence: " + ", ".join(record["draft"]["evidence_ids"]) + "\n")
        atomic(directory / "SKILL.md", text.encode())
        existing = proposals.find_skill_proposal(self.root, name)
        if existing and existing.status == "approved":
            valid, errors, _ = proposals._validate_skill_proposal_dir(memory_dir=self.root, skill_name=name)
            return {"submitted": valid, "errors": errors}
        result = proposals.submit_autoskill_proposal(memory_dir=self.root, skill_name=name, cluster_hash=key,
            source_observation_ids=record["draft"]["evidence_ids"], rationale=record["draft"]["applicability"],
            skills_dir=self.root / "skills", workspace_dir=self.root.parent, project_id=record["project"])
        return result

    def publish_skill(self, key, record):
        name = self.name(key)
        result = self.stage_skill(key, record)
        if not result["submitted"]:
            raise ValueError("Upstream validation failed: " + json.dumps(result, ensure_ascii=False))
        proposal = proposals.find_skill_proposal(self.root, name)
        if proposal.status != "approved":
            result = proposals.approve_skill_proposal(self.root, name, skills_dir=self.root / "skills",
                workspace_dir=self.root.parent, scope="workspace", approved_by="user", exact=True,
                expected_updated_at=proposal.updated_at)
            if not result["approved"]:
                raise ValueError("Upstream publication failed: " + json.dumps(result, ensure_ascii=False))
        path = self.root / "skills" / name / "SKILL.md"
        staged = self.root / "autoskills" / "proposals" / name / "SKILL.md"
        if path.read_bytes() != staged.read_bytes():
            raise ValueError("Installed skill differs from reviewed snapshot")
        return {"skill_path": str(path), "skill_sha256": digest(path.read_bytes()), "upstream_proposal": name}

    def approve(self, key: str, *, reviewer: str, expected_hash: str, explanation: str):
        if not reviewer.strip() or not explanation.strip() or expected_hash != key:
            raise ValueError("Approval must bind a concrete hash and human rationale")
        with StateStore(self.root.parent).publication_gate(), self.lock:
            record = self.read(key)
            path = self.root / "approved" / f"{key}.json"
            if path.exists():
                old = self.read(key, "approved")
                if old["status"] == "approved":
                    return old
                raise ValueError("Inactive version requires an explicit rollback decision")
            screen = self._screen(key)
            if not screen["passed"]:
                raise ValueError("External validation required before promotion: " + "; ".join(screen["errors"]))
            review = self.logic_review(key)
            self.require_grounded(key, screen["evidence"], review["evidence"] if review else None)
            previous = record.get("supersedes")
            if previous and self.read(previous, "approved")["status"] != "approved":
                raise ValueError("Update target is no longer active")
            if previous:
                for decision_path in (self.root / "approved").glob("*.json"):
                    existing = self.read(decision_path.stem, "approved")
                    if existing["status"] == "approved" and existing.get("supersedes") == previous:
                        raise ValueError("Update target already has a committed successor")
            decision = {"proposal": key, "project": record["project"], "status": "approved",
                        "approved_by": reviewer, "approved_at": time.time(), "approval_hash": key,
                        "approval_rationale": explanation, "screen": screen["evidence"],
                        "logic_review": review["evidence"]}
            if record["draft"]["kind"] == "skill":
                decision.update(self.publish_skill(key, record))
            # The approval file is the retrieval commit point. Draft bytes stay immutable.
            atomic(path, canonical(decision))
            if previous:
                self.deactivate(previous, "superseded", f"Replaced by {key}")
            self.event(key, "approved", reviewer=reviewer, rationale=explanation)
            self.rebuild_index()
            return self.read(key, "approved")

    def deactivate(self, key, status, reason):
        path = self.root / "approved" / f"{key}.json"
        decision = json.loads(path.read_bytes())
        decision.update(status=status, inactive_at=time.time(), inactive_reason=reason)
        atomic(path, canonical(decision))
        skill = self.root / "skills" / self.name(key) / "SKILL.md"
        if skill.exists():
            skill.unlink()
        self.event(key, status, reason=reason)

    def revoke(self, key: str, reason: str):
        if not reason.strip():
            raise ValueError("Revocation needs a reason")
        with self.lock:
            self.read(key, "approved")
            self.deactivate(key, "revoked", reason)
            self.rebuild_index()

    def rollback(self, key: str, *, reviewer: str, explanation: str):
        if not reviewer.strip() or not explanation.strip():
            raise ValueError("Rollback requires explicit human review")
        with StateStore(self.root.parent).publication_gate(), self.lock:
            record = self.read(key, "approved")
            if record["status"] == "revoked":
                raise ValueError("Revoked experience requires a new evidence-backed proposal")
            if not self.screen(key)["passed"]:
                raise ValueError("Rollback evidence no longer valid")
            decision = json.loads((self.root / "approved" / f"{key}.json").read_bytes())
            # Validate the snapshot before withdrawing any active version.
            if record["draft"]["kind"] == "skill":
                source = self.root / "autoskills" / "proposals" / self.name(key) / "SKILL.md"
                data = source.read_bytes()
                if digest(data) != decision["skill_sha256"]:
                    raise ValueError("Rollback snapshot changed")
            for path in (self.root / "approved").glob("*.json"):
                other = self.read(path.stem, "approved")
                if other["status"] != "approved" or path.stem == key:
                    continue
                ancestor = other.get("supersedes")
                while ancestor:
                    if ancestor == key:
                        self.deactivate(path.stem, "superseded", f"Human rollback to {key}")
                        break
                    ancestor = self.read(ancestor).get("supersedes")
            if record["draft"]["kind"] == "skill":
                atomic(self.root / "skills" / self.name(key) / "SKILL.md", data)
            decision.update(status="approved", rollback_by=reviewer, rollback_reason=explanation)
            atomic(self.root / "approved" / f"{key}.json", canonical(decision))
            self.event(key, "rollback", reviewer=reviewer, rationale=explanation)
            self.rebuild_index()

    def timeline(self, project: str, limit=20) -> list[dict]:
        """L1/L4, explicitly untrusted; not part of long-term retrieval."""
        self.require_scope(project)
        items = [dict(self.read(p.stem), id=p.stem, trust="candidate_unverified")
                 for p in (self.root / "candidates").glob("*.json")]
        recent = sorted((r for r in items if r["project"] == project), key=lambda r: r["batch"])[-limit:]
        for item in recent:
            review = self.logic_review(item["id"])
            if review:
                item["logic_review"] = {key: review[key] for key in ("evidence", "status", "human_action")
                                        if key in review}
        return recent

    def rebuild_index(self):
        with sqlite3.connect(self.root / "index.sqlite") as db:
            db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS records USING fts5(id UNINDEXED, project UNINDEXED, text)")
            db.execute("DELETE FROM records")
            records = [(path.stem, self.read(path.stem, "approved"))
                       for path in sorted((self.root / "approved").glob("*.json"))]
            replaced = set()
            for _key, record in records:
                if record["status"] != "approved":
                    continue
                ancestor = record.get("supersedes")
                while ancestor:
                    replaced.add(ancestor)
                    ancestor = self.read(ancestor).get("supersedes")
            for key, record in records:
                # Also handles a crash after new approval commit but before old deactivation.
                if key in replaced:
                    continue
                if record["status"] == "approved":
                    db.execute("INSERT INTO records VALUES(?,?,?)", (key, record["project"],
                               record["draft"]["title"] + " " + record["draft"]["content"]))

    def search(self, query: str, project: str, limit=8) -> list[dict]:
        self.require_scope(project)
        with self.lock:
            self.rebuild_index()
            with sqlite3.connect(self.root / "index.sqlite") as db:
                terms = [t for t in query.split() if t.isalnum()]
                if not terms:
                    rows = db.execute("SELECT id FROM records WHERE project=? LIMIT ?", (project, limit))
                else:
                    rows = db.execute("SELECT id FROM records WHERE records MATCH ? AND project=? ORDER BY rank LIMIT ?",
                                      (" OR ".join('"' + t + '"' for t in terms), project, limit))
                return [self.read(row[0], "approved") for row in rows]
