"""Translate a passed replay into an evidence-linked new experiment admission review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import FileLock
from pydantic import Field

from .models import StrictModel
from .onboard_replay import resolve_inside
from .storage import Evidence, atomic, canonical, digest
from .workers import CodexWorker


class CandidateAdmission(StrictModel):
    existing_candidate_command_ready: bool
    existing_frozen_scoring_command_ready: bool
    scope: str
    proposed_axes: list[str] = Field(max_length=4)
    capabilities_to_preserve: list[str]
    required_artifacts: list[str]
    missing_gates: list[str]


def assess(root: Path, worker=None) -> dict:
    root = root.resolve(strict=True)
    with FileLock(str(root / ".onboarding.lock"), timeout=0):
        return _assess(root, worker or CodexWorker(root))


def _assess(root: Path, worker) -> dict:
    state = json.loads((root / "onboarding.json").read_bytes())
    evidence = Evidence(root)
    plan = evidence.get(state["plan"])["content"]["plan"]
    execution_id = json.loads((root / "replay-execution.json").read_bytes())["evidence"]
    replay = evidence.get(execution_id)["content"]
    if replay["phase"] != "replay_executed":
        raise ValueError("A new experiment admission review requires an executed historical replay")
    workspace = Path(replay["workspace"])
    inspected = []
    for file, expected in replay["outputs"].items():
        raw = resolve_inside(workspace, file).read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Replay evidence mismatch: {file}")
        result = json.loads(raw)
        inspected.append({"path": file, "sha256": expected, "fields": sorted(result),
                          "observed": {key: result[key] for key in (
                              "passed", "status", "aggregate", "stage_counts", "new_fits"
                          ) if key in result}})
    source_files = evidence.get(state["sources"])["content"]["cards"]
    entrypoints = set(plan["baseline_entrypoints"] + plan["independent_evaluation_entrypoints"])
    cards = [card for card in source_files if card["path"] in entrypoints]
    proposal, metadata = worker.run("onboard-admission", {
        "draft": plan, "audited_entrypoints": cards, "replayed_evidence": inspected,
        "requirements": "Review readiness for a new experiment and respond in Chinese. Distinguish a historical "
        "verifier whose assertions depend on saved historical results from a frozen scorer accepting new candidate "
        "predictions. Approve a capability only if the existing repository input proves its reusable contract. "
        "Describe prerequisites for machine-checked subject-level folds, candidate-controlled parameters, immutable "
        "scoring, independent checks, and output manifests. Suggest only experiments consistent with project goals. "
        "No experiments are executed in this assessment, and no independent holdout may be fabricated.",
    }, CandidateAdmission)
    proposal_id = evidence.put("onboarding_candidate_admission", {"proposal": proposal.model_dump(), "worker": metadata},
                               [execution_id, state["plan"], state["review"]])
    # A worker opinion cannot promote the replay itself to a runnable generic adapter.
    record = {"phase": "candidate_contract_pending", "evidence": proposal_id, **proposal.model_dump()}
    atomic(root / "candidate-admission.json", canonical(record))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
