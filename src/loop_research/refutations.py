"""Review empirical rejections and apply them only to the measured conditions."""
from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Literal

from filelock import FileLock
from pydantic import Field

from .command_candidate import candidate_parts
from .models import StrictModel
from .storage import atomic, canonical, digest


class RefutationAssessment(StrictModel):
    verdict: Literal["supported", "inconclusive", "rejected"]
    explanation: str = Field(min_length=10, max_length=4000)
    limitations: list[str] = Field(min_length=1, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=32)


def index(engine):
    return engine.root / "direction-refutations"


def approval_lock(engine):
    folder = index(engine)
    folder.mkdir(parents=True, exist_ok=True)
    return FileLock(str(folder / ".approval.lock"))


def condition(parameters, model, name):
    return {"model_sha256": digest(canonical(model)), "fixed_parameters": {k: v for k, v in parameters.items() if k != name},
            "tested_value": parameters[name]}


def active_rule(engine, code):
    if not engine.command:
        return None
    for path in sorted((index(engine) / "active").glob("*.json")):
        key = json.loads(path.read_bytes())["evidence"]
        record = engine.evidence.get(key)
        rule = record["content"]
        if (record["kind"] != "direction_refutation_approval" or rule["protocol"] != engine.references()["protocol"]
                or rule["project"] != engine.store.read().project_id):
            continue
        parameters, model = candidate_parts(code, engine.command)
        current = condition(parameters, model, rule["parameter"])
        if any(canonical(current) == canonical(row["condition"]) for row in rule["pairs"]):
            engine.evidence.verify_tree(key)
            return key
    return None


def measured_code(engine, key, result):
    for parent in engine.evidence.get(key)["parents"]:
        content = engine.evidence.get(parent)["content"]
        code = content.get("code")
        if code is not None and digest(code.encode()) == result["code_sha256"]:
            return code
    raise ValueError("Measurement must point to its actual candidate or control code")


def measured_pairs(engine, measurement_ids):
    from .followup import history

    pending = set(measurement_ids)
    groups = []
    for batch_id, batch in history(engine):
        decision_path = engine.root / "decisions" / f"{batch['batch']:06d}.json"
        if not decision_path.is_file():
            continue
        decision_id = json.loads(decision_path.read_bytes())["evidence"]
        decision = engine.evidence.get(decision_id)["content"]
        if decision.get("snapshot") != batch_id:
            raise ValueError("Refutation requires the decided batch snapshot")
        for task in batch["tasks"]:
            evidence_id = task.get("evidence")
            if evidence_id in pending:
                groups.append((task, batch, decision_id))
                pending.remove(evidence_id)
        if not pending:
            return groups
    raise ValueError("Every refutation measurement must belong to a decided batch")


def propose(engine, measurement_ids):
    if not engine.command or len(set(measurement_ids)) != len(measurement_ids) or not 2 <= len(measurement_ids) <= 8:
        raise ValueError("A direction refutation requires two to eight distinct measured command candidates")
    refs = engine.references()
    pairs, targets, papers, parents = [], set(), {}, [refs["protocol"], refs["knowledge"]]
    for task, batch, decision_id in measured_pairs(engine, measurement_ids):
        target = task["hypothesis"]["machine_target"] or ""
        if task["status"] != "succeeded" or not target.startswith("parameter."):
            raise ValueError("Use successful, isolated parameter interventions rather than failed jobs")
        name = target.removeprefix("parameter.")
        measurement, control = task["evidence"], batch["baseline"]
        candidate, baseline = (engine.evidence.get(key)["content"] for key in (measurement, control))
        engine.verify_validation(candidate, control)
        if candidate["split"] != "validation" or not engine.dominates(baseline["metrics"], candidate["metrics"]):
            raise ValueError("Every candidate must be strictly worse under its matched development control")
        changed, candidate_model = candidate_parts(measured_code(engine, measurement, candidate), engine.command)
        fixed, baseline_model = candidate_parts(measured_code(engine, control, baseline), engine.command)
        differences = {key for key in changed if canonical(changed[key]) != canonical(fixed[key])}
        if differences != {name} or candidate_model != baseline_model:
            raise ValueError("A measured pair must change exactly the nominated parameter")
        targets.add(name)
        pairs.append({"condition": condition(changed, candidate_model, name), "control_value": fixed[name],
                      "candidate": measurement, "control": control, "metrics": candidate["metrics"],
                      "control_metrics": baseline["metrics"], "batch_decision": decision_id})
        parents.extend([decision_id, measurement, control])
        for key in task["hypothesis"].get("source_ids", []):
            record = engine.evidence.get(key)
            if record["kind"] == "paper_source":
                paper = record["content"]
                papers[key] = {"title": paper.get("title"), "abstract": str(paper.get("abstract") or "")[:2200]}
    if len(targets) != 1 or len({canonical(row["condition"]) for row in pairs}) != len(pairs):
        raise ValueError("A direction refutation needs distinct measured conditions for one parameter")
    parameter = next(iter(targets))
    parents = list(dict.fromkeys([*parents, *papers]))
    record = {"protocol": refs["protocol"], "project": engine.store.read().project_id, "parameter": parameter, "pairs": pairs}
    key = engine.evidence.put("direction_refutation_proposal", record, parents)
    path = index(engine) / "proposals" / f"{key}.json"
    payload = {"parameter": parameter, "paired_measurements": pairs, "evidence_ids": parents, "papers": papers,
               "confirmed_context": {"evidence_id": refs["knowledge"], "text": engine.project.knowledge},
               "requirements": "Independently check the numeric decreases, the matched control and the scope of the claim. "
                               "Use the original scientific sources if provided. Separate observed effects from explanations. "
                               "The proposed rule covers only the enumerated conditions. Do not infer a wider range or causality. "
                               "Use inconclusive for a material ambiguity. Cite all candidate, control and confirmed context IDs."}
    required = {refs["knowledge"], *(row[field] for row in pairs for field in ("candidate", "control"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.with_suffix(".lock"))):
        if path.is_file():
            return json.loads(path.read_bytes())
        assessment, metadata = engine.reviewer.run("refutation-attribution-review", payload, RefutationAssessment)
        cited_correctly = required <= set(assessment.evidence_ids) <= set(parents)
        review = engine.evidence.put("direction_refutation_attribution", {
            "assessment": assessment.model_dump(), "worker": metadata, "citations_verified": cited_correctly}, [key, *parents])
        queue = {"proposal": key, "review": review, "status": "awaiting_human",
                 "eligible_for_approval": cited_correctly and assessment.verdict == "supported"}
        atomic(path, canonical(queue))
        return queue


def decide(engine, proposal_id, reviewer, rationale, decision):
    refs = engine.references()
    if not reviewer.strip() or not rationale.strip():
        raise ValueError("A direction refutation decision requires a reviewer and rationale")
    proposal = engine.evidence.get(proposal_id)
    if (proposal["kind"] != "direction_refutation_proposal" or proposal["content"]["protocol"] != refs["protocol"]
            or proposal["content"]["project"] != engine.store.read().project_id):
        raise ValueError("Expected a direction refutation proposal ID")
    gate = engine.store.publication_gate() if decision == "approve" else nullcontext()
    with approval_lock(engine), gate:
        queue = json.loads((index(engine) / "proposals" / f"{proposal_id}.json").read_bytes())
        if queue["status"] != "awaiting_human":
            raise ValueError("This proposal already has a final decision")
        review_id = queue["review"]
        assessment = engine.evidence.get(review_id)
        if assessment["kind"] != "direction_refutation_attribution" or proposal_id not in assessment["parents"]:
            raise ValueError("Proposal and independent review do not match")
        engine.evidence.verify_tree(review_id)
        reviewed = assessment["content"]
        if decision == "approve" and not (reviewed["citations_verified"] and reviewed["assessment"]["verdict"] == "supported"):
            raise ValueError("Unsupported attribution remains pending for additional evidence")
        kind = "direction_refutation_approval" if decision == "approve" else "direction_refutation_disposition"
        key = engine.evidence.put(kind, {**proposal["content"], "decision": decision,
                                  "reviewer": reviewer, "rationale": rationale}, [proposal_id, review_id])
        if decision == "approve":
            atomic(index(engine) / "active" / f"{key}.json", canonical({"evidence": key}))
        if decision != "defer":
            queue["status"] = "approved" if decision == "approve" else "rejected"
        queue["decision"] = key
        atomic(index(engine) / "proposals" / f"{proposal_id}.json", canonical(queue))
        return queue
