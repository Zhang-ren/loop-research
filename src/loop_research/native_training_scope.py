"""Qualify a workflow-authored training refactor under the original frozen scorer."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field

from .models import StrictModel
from .command_candidate import CODE_LIMIT
from .native_adaptation import AdapterSource
from .native_study_runtime import validate_mutable_sources
from .onboard_native import NativeReview, checked_stage_views, failure_context, read_json, required_inputs, run_recipe
from .onboard_replay import resolve_inside
from .sealed_stage import file_hash
from .storage import Evidence, atomic, canonical, digest
from .workers import WorkerFailure

DIRECTORY = "loop-training-scope"

CONTRACT = """
Create a behavior-preserving training refactor in the isolated study bundle. Keep the scorer,
its imports, score metadata, data, all arguments and selected metrics unchanged. Only change
one training stage's Python entrypoint. The new driver takes exactly its existing argv and
writes exactly its declared outputs in the same directory. Move baseline training behavior
into named modules invisible to the scorer, so later candidate source changes can act across
multiple files. Offer genuine training abstractions such as loss, initialization, optimization,
or stochastic regularization policy. Do not insert dummy files or a scientific improvement
into the measured baseline. Reuse upstream functions or audited adapter functions when
possible; freeze any common helper used by scoring. Choose modules with the same inference
and model output contract that the original scorer can still read. Return new distinct Python
files directly under loop-training-scope. Do not overwrite or add anything to upstream.
Only propose an honest blocker if no behavior-preserving candidate scope is reasonably
possible under the requested protocol. This actor does not run or report measured experiments.
"""


class TrainingScopeDraft(StrictModel):
    stage: str
    driver: AdapterSource | None
    modules: list[AdapterSource] = Field(max_length=6)
    explanation: str
    blockers: list[str] = Field(max_length=6)


def scoped_recipe(recipe, draft, minimum):
    views = checked_stage_views(recipe)
    eligible = {stage.name for stage in recipe.stages[:-1]}
    if not views or draft.stage not in eligible or not draft.driver or len(draft.modules) < minimum:
        raise ValueError("The refactor needs a measured isolated training stage and the required number of source modules")
    generated = [draft.driver, *draft.modules]
    names = [item.path for item in generated]
    if len(set(names)) != len(names) or set(names).intersection(recipe.source_files + recipe.input_files):
        raise ValueError("The refactor must write only distinct new paths")
    if sum(len(item.content.encode()) for item in draft.modules) > CODE_LIMIT:
        raise ValueError("Mutable training modules exceed the candidate size limit")
    for item in generated:
        path = Path(item.path)
        if path.parent.as_posix() != DIRECTORY or path.suffix != ".py":
            raise ValueError(f"Invalid generated training module path: {item.path}")
        try:
            compile(item.content, item.path, "exec")
        except SyntaxError as exc:
            raise ValueError(f"Invalid generated Python in {item.path}:{exc.lineno}") from exc
    view = views[draft.stage]
    views[draft.stage] = view.model_copy(update={"source_files": [*view.source_files, *names]})
    original_stage = next(stage for stage in recipe.stages if stage.name == draft.stage)
    new_stage = original_stage.model_copy(update={"script": draft.driver.path})
    scoped = recipe.model_copy(update={
        "stages": [new_stage if stage.name == draft.stage else stage for stage in recipe.stages],
        "source_files": [*recipe.source_files, *names],
        "stage_views": [views[stage.name] for stage in recipe.stages],
    })
    validate_mutable_sources(scoped, names[1:], [draft.stage])
    return scoped, generated


def seal_sources(target, project, cards, generated):
    result = dict(cards)
    for name, card in cards.items():
        raw = resolve_inside(project, name).read_bytes()
        if digest(raw) != card["sha256"]:
            raise ValueError(f"Previously reviewed source changed: {name}")
        path = resolve_inside(target, name, existing=False)
        if not path.exists():
            atomic(path, raw)
        if file_hash(path) != card["sha256"]:
            raise ValueError(f"Training scope copy changed: {name}")
    for item in generated:
        raw = item.content.encode()
        sha = digest(raw)
        path = resolve_inside(target, item.path, existing=False)
        if not path.exists():
            atomic(path, raw)
        if file_hash(path) != sha:
            raise ValueError(f"Generated training module changed: {item.path}")
        result[item.path] = {"path": item.path, "sha256": sha, "excerpt": item.content}
    return result


def prepare_training_scope(root, folder, project, interpreter, baseline, cards, certificate_ref, worker, minimum):
    store = Evidence(root)
    certificate = store.get(certificate_ref)["content"]
    state_path = folder / "training-scope.json"
    state = read_json(state_path) if state_path.exists() else {}
    identity = {"baseline": certificate_ref, "minimum": minimum}
    if state and any(state.get(key) != value for key, value in identity.items()):
        raise ValueError("Existing refactor qualification belongs to another baseline or scope")

    def save(**fields):
        state.update(identity, **fields)
        atomic(state_path, canonical(state))

    originals = {name: resolve_inside(project, name).read_text(encoding="utf-8")
                 for name in baseline.source_files if name.endswith(".py")}
    if sum(len(value.encode()) for value in originals.values()) > 100_000:
        raise ValueError("Training sources exceed the reviewed refactor input limit")
    numeric = certificate["metrics"]
    payload = {"original_recipe": baseline.model_dump(), "original_sources": originals,
               "minimum_mutable_files": minimum, "original_development_metrics": numeric, "requirements": CONTRACT}
    failure = state.get("failure")
    for _ in range(3):
        parents = [certificate_ref, *([failure] if failure else [])]
        try:
            if state.get("draft"):
                draft_ref = state["draft"]
                draft = TrainingScopeDraft.model_validate(store.get(draft_ref)["content"]["draft"])
            else:
                draft, metadata = worker.run("native-training-scope", {
                    **payload, "prior_failure": store.get(failure)["content"] if failure else None,
                }, TrainingScopeDraft)
                draft_ref = store.put("native_training_scope_draft", {"draft": draft.model_dump(), "worker": metadata}, parents)
                save(draft=draft_ref, review=None)
            parents.append(draft_ref)
            if draft.blockers:
                raise ValueError("Workflow could not isolate training modules: " + " ".join(draft.blockers))
            refined, generated = scoped_recipe(baseline, draft, minimum)
            if state.get("review"):
                review_ref = state["review"]
            else:
                review, metadata = worker.run("native-training-scope-review", {
                    **payload, "draft": draft.model_dump(), "refined_recipe": refined.model_dump(),
                    "requirements": "Independently verify an equivalent baseline. " + CONTRACT +
                                    " Check seed, training math, interface, state serialization and output paths.",
                }, NativeReview)
                review_ref = store.put("native_training_scope_review",
                    {"review": review.model_dump(), "worker": metadata}, parents)
                if not review.accepted:
                    raise ValueError("Independent refactor review rejected: " + " ".join(review.findings))
                save(review=review_ref)
            parents.append(review_ref)
            workspace = folder / "training-scope" / draft_ref / "inputs"
            workspace.mkdir(parents=True, exist_ok=True)
            sealed = seal_sources(workspace, project, cards, generated)
            input_hashes = store.get(certificate["recipe"])["content"].get("input_sha256") or {}
            _, missing = required_inputs(project, None, baseline.input_files)
            if missing:
                raise ValueError("Measured development inputs are missing: " + ", ".join(missing))
            receipts, measured = run_recipe(workspace, workspace.parent, interpreter, project, refined, sealed,
                                            input_hashes=input_hashes)
            if measured is None:
                raise ValueError(f"Refactored training stage failed: {failure_context(receipts, workspace)}")
            if measured != numeric:
                raise ValueError(f"Refactored training did not reproduce the original development score: {measured}")
            stage_refs = [store.put("native_stage_receipt", receipt, parents) for receipt in receipts]
            proof = store.put("native_training_scope_verified", {
                "metrics": measured, "paths": [item.path for item in draft.modules], "stage": draft.stage,
                "workspace": str(workspace), "stages": stage_refs,
            }, [*parents, *stage_refs])
            save(verified=proof)
            return workspace, refined, sealed, draft, proof
        except (OSError, ValueError, WorkerFailure) as exc:
            failure = store.put("native_training_scope_rejected", {"reason": str(exc)}, parents)
            save(draft=None, review=None, failure=failure)
    raise ValueError(store.get(failure)["content"]["reason"])
