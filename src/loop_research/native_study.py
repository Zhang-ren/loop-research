"""Admit a verified native baseline to the local candidate state machine."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from filelock import FileLock
from pydantic import Field

from .engine import Research
from .models import CommandExperiment, Metric, MetricPolicy, Project, StrictModel
from .native_study_runtime import NativeAxis, apply_axes, validate_mutable_sources
from .native_source_axes import validate_source_scope
from .native_training_scope import prepare_training_scope
from .onboard_native import (NativeRecipe, NativeReview, ScoreInputBinding, StageDirectory, StageView, inside,
                             required_inputs, snapshot_files, validate_recipe)
from .onboard_replay import resolve_inside
from .onboarding import PYTHON_MAIN, SUPPORT_EXCERPT, available_paths
from .preflight import inspect, issue
from .sealed_stage import Stage, file_hash, stage_command
from .storage import Evidence, atomic, canonical, digest
from .workers import CodexWorker, WorkerFailure

ADMISSION_SOURCE_LIMIT = 100_000
ADMISSION_FILE_LIMIT = 32_000


class StudyDraft(StrictModel):
    explanation: str
    original_inputs: list[str] = Field(max_length=16)
    projected_inputs: list[str] = Field(max_length=16)
    additional_source_files: list[str] = Field(max_length=20)
    preparation_stages: list[Stage] = Field(max_length=4)
    replaced_preparation: list[str] = Field(default_factory=list, max_length=4)
    working_directories: list[StageDirectory] = Field(max_length=4)
    split_builder: str = Field(max_length=24_000)
    axes: list[NativeAxis] = Field(max_length=12)
    mutable_sources: list[str] = Field(default_factory=list, max_length=6)
    mutable_stages: list[str] = Field(default_factory=list, max_length=5)
    stage_views: list[StageView] = Field(default_factory=list, max_length=6)
    score_input_bindings: list[ScoreInputBinding] = Field(default_factory=list, max_length=16)
    primary_metric: str
    direction: Literal["min", "max"]
    blockers: list[str] = Field(max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=8)


def validate_axes(recipe: NativeRecipe, axes: list[NativeAxis], preparation_names: set[str], *, allow_empty=False):
    if (not axes and not allow_empty) or len({axis.name for axis in axes}) != len(axes):
        raise ValueError("The study requires distinct native parameter axes")
    bindings = set()
    trains = {stage.name for stage in recipe.stages[:-1]} - preparation_names
    for axis in axes:
        if (len(set(axis.choices)) < 2 or axis.baseline not in axis.choices
                or any(type(value) is not type(axis.baseline) for value in axis.choices)):
            raise ValueError(f"Invalid choices for native parameter {axis.name}")
        if axis.source_binding:
            identifiers = validate_source_scope(recipe, axis, trains)
            selector = axis.source_binding
            binding = selector.path, selector.kind, selector.target, tuple(axis.key_path)
        elif axis.config_file:
            if ((axis.stage is not None and axis.stage not in trains) or axis.argument or not axis.key_path
                    or axis.config_file not in recipe.source_files
                    or Path(axis.config_file).suffix not in {".yml", ".yaml", ".json"}):
                raise ValueError("Config axes must point to one existing scalar in a sealed config")
            identifiers = axis.key_path
            binding = axis.config_file, tuple(axis.key_path)
        else:
            if axis.stage not in trains or axis.key_path or not re.fullmatch(r"--[a-zA-Z][\w-]*", axis.argument or ""):
                raise ValueError("CLI axes must refer to a training stage's existing long option")
            identifiers = [axis.argument.removeprefix("--")]
            binding = axis.stage, axis.argument
        tokens = {token for name in identifiers for token in re.split(r"[_-]", name.lower())}
        if (any(name.casefold() == "split" for name in identifiers)
                or tokens & {"test", "eval", "evaluate", "validation", "holdout", "seed", "fold", "folds", "scorer",
                             "score", "metric", "metrics", "label", "target", "dataset", "data", "output", "path"}):
            raise ValueError(f"Research axis overlaps a frozen protocol field: {axis.name}")
        if binding in bindings:
            raise ValueError(f"Duplicate native parameter binding: {axis.name}")
        bindings.add(binding)


def recipe_for(project, baseline, draft, plan, cards):
    if baseline.attestations or baseline.readonly_directories or baseline.seed_paths:
        raise ValueError("External baseline mounts need a standalone sealed data projection for candidate runs")
    names = {stage.name for stage in draft.preparation_stages}
    if not {item.stage for item in draft.working_directories} <= names:
        raise ValueError("Existing train and score working directories must stay unchanged")
    count = len(draft.replaced_preparation)
    if count and (count > len(baseline.stages) - 2
                  or draft.replaced_preparation != [stage.name for stage in baseline.stages[:count]]):
        raise ValueError("Only an initial preparation prefix may be replaced by the sealed projection")
    if count and any(not any(inside(path, output) for path in draft.projected_inputs)
                     for output in baseline.stages[count - 1].outputs):
        raise ValueError("Projection must supply the outputs of the last replaced preparation stage")
    existing_views = {item.stage: item for item in baseline.stage_views if item.stage not in draft.replaced_preparation}
    supplied_views = {item.stage: item for item in draft.stage_views}
    if any(stage in supplied_views and item != supplied_views[stage] for stage, item in existing_views.items()):
        raise ValueError("The study must preserve the measured stage isolation")
    if baseline.score_input_bindings and draft.score_input_bindings not in ([], baseline.score_input_bindings):
        raise ValueError("The study cannot replace a measured scoring input binding")
    recipe = NativeRecipe.model_validate({**baseline.model_dump(),
        "stages": [*draft.preparation_stages, *baseline.stages[count:]],
        "working_directories": [*(item for item in baseline.working_directories
                                   if item.stage not in draft.replaced_preparation), *draft.working_directories],
        "source_files": list(dict.fromkeys([*baseline.source_files, *draft.additional_source_files,
                                           *(stage.script for stage in draft.preparation_stages)])),
        "input_files": draft.projected_inputs,
        "stage_views": list(existing_views.values()) + [item for item in draft.stage_views if item.stage not in existing_views],
        "score_input_bindings": draft.score_input_bindings or baseline.score_input_bindings,
    })
    preparation = [stage.script for stage in draft.preparation_stages]
    for name in preparation:
        if (name not in cards or not name.endswith(".py")
                or (name not in plan["baseline_entrypoints"]
                    and not PYTHON_MAIN.search(resolve_inside(project, name).read_bytes()))):
            raise ValueError(f"Missing original standalone preparation entrypoint: {name}")
    validate_recipe(recipe, project, None, {
        **plan, "baseline_entrypoints": [*plan["baseline_entrypoints"], *preparation],
    }, cards)
    if names.intersection(draft.mutable_stages):
        raise ValueError("Data preparation stages cannot contain structural changes")
    validate_mutable_sources(recipe, draft.mutable_sources, draft.mutable_stages)
    validate_axes(recipe, draft.axes, names, allow_empty=bool(draft.mutable_sources))
    if any(axis.source_binding and {axis.source_binding.path, axis.source_binding.default_source}
           .intersection(draft.mutable_sources) for axis in draft.axes):
        raise ValueError("A Python parameter source and its default cannot also be editable by structural candidates")
    if not draft.split_builder.strip() or not draft.original_inputs or not draft.projected_inputs:
        raise ValueError("The study needs a builder and explicit source and staged inputs")
    if draft.primary_metric not in {item.name for item in recipe.metrics}:
        raise ValueError("The primary metric must be produced by the original scorer")
    return recipe


def prepare_bundle(folder, interpreter, draft, recipe, project, files, cards, baseline_ref, source_hashes):
    with TemporaryDirectory(prefix="projection-", dir=folder) as temporary:
        scratch = Path(temporary)
        for name, source in files.items():
            target = resolve_inside(scratch, name, existing=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in recipe.source_files:
            original = resolve_inside(project, name)
            target = resolve_inside(scratch, name, existing=False)
            if target.exists() or file_hash(original) != cards[name]["sha256"]:
                raise ValueError(f"Projection source overlaps a data input or changed after review: {name}")
            atomic(target, original.read_bytes())
        script = "loop-project-inputs.py"
        atomic(scratch / script, draft.split_builder.encode())
        before = snapshot_files(scratch)
        if any(before.get(name) != sha for name, sha in source_hashes.items()):
            raise ValueError("A source dataset changed while materializing the projection")
        command = stage_command(interpreter, scratch, project, script, [], working_directory=".")
        with (folder / "projection.stdout.log").open("wb") as out, (folder / "projection.stderr.log").open("wb") as err:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=300)
        if result.returncode:
            tail = (folder / "projection.stderr.log").read_text(errors="replace")[-1400:]
            raise ValueError(f"Data projection exited {result.returncode}: {tail}")
        after = snapshot_files(scratch)
        if any(after.get(name) != sha for name, sha in before.items()):
            raise ValueError("Data projection changed an original input")
        manifest = json.loads(resolve_inside(scratch, "split.json").read_bytes())
        scopes = ["validation"] if "folds" in manifest and not manifest.get("heldout") else ["validation", "test"]
        expected = {"split.json", *(f"views/{scope}/{name}" for scope in scopes for name in recipe.input_files)}
        produced = set(after) - set(before)
        if produced != expected:
            missing = sorted(expected - produced)[:20]
            extra = sorted(produced - expected)[:20]
            raise ValueError(f"Projection must emit exactly split.json and the admitted input files for each data view. "
                             f"Missing: {missing}. Unexpected: {extra}")
        declaration = draft.model_dump()
        for axis in declaration["axes"]:
            if axis.get("source_binding") is None:
                axis.pop("source_binding", None)
        identity = digest(canonical([declaration, baseline_ref, source_hashes, after]))
        repository = folder / "bundles" / identity
        repository.mkdir(parents=True, exist_ok=True)
        for name in recipe.source_files:
            original = resolve_inside(project, name)
            if file_hash(original) != cards[name]["sha256"]:
                raise ValueError(f"Audited original source changed: {name}")
            atomic(resolve_inside(repository, f"native/{name}", existing=False), original.read_bytes())
        for name in sorted(expected):
            atomic(resolve_inside(repository, name, existing=False), resolve_inside(scratch, name).read_bytes())
        view_sha = {scope: {name: after[f"views/{scope}/{name}"] for name in recipe.input_files} for scope in scopes}
        preparation = [stage.name for stage in draft.preparation_stages]
        bindings = {"preparation_stages": preparation} if any(axis.source_binding for axis in draft.axes) else {}
        atomic(repository / "native-study.json", canonical({
            "recipe": recipe.model_dump(), "axes": declaration["axes"],
            "mutable_sources": draft.mutable_sources, "mutable_stages": draft.mutable_stages,
            "view_sha256": view_sha, "original_input_sha256": source_hashes, "origin": baseline_ref,
            "native_source_sha256": {name: cards[name]["sha256"] for name in recipe.source_files},
            **bindings,
        }))
        with TemporaryDirectory(prefix="check-defaults-", dir=folder) as trial:
            trial = Path(trial)
            shutil.copytree(repository / "native", trial, dirs_exist_ok=True)
            apply_axes(trial, recipe, draft.axes, {axis.name: axis.baseline for axis in draft.axes}, preparation)
    return repository


def commit_bundle(repository: Path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null",
                                        *args], text=True, timeout=90).strip()
    git("init", "-q", "--template=")
    git("add", "--", "native", "views", "split.json", "native-study.json")
    if git("diff", "--cached", "--name-only"):
        git("-c", "user.name=LOOP Research", "-c", "user.email=local@invalid", "commit", "-qm",
            "Seal native candidate and evaluation protocol")
    return git("rev-parse", "HEAD")


PROJECTION_CONTRACT = (
    "Generate an admission draft only. Keep measured train and score scripts and all arguments identical; "
    "the controller binds axes to their existing scalar defaults. Prefer 2-3 distinct axes with documented reasonable "
    "options. If no CLI or config binding exists but the sealed Python source already has an equivalent train "
    "control, declare source_binding. Set path, kind and target. For kind=assignment, target names a top-level "
    "assignment and key_path identifies a scalar within its literal dictionaries, or is empty for the scalar "
    "assignment itself. For kind=keyword, target is the exact existing callable expression and key_path contains "
    "one keyword. If omitted by the original call, give default_source and default_callable, for example "
    "Model.__init__, resolving to the original literal default in the training view. Set config_file and argument "
    "to null and stage to the admitted training stage. The measured default preserves the original bytes. "
    "Python keyword bindings are training-only. A model-literal binding shared with scoring is allowed solely "
    "when the scorer uses that literal to reconstruct the trained model or trace its configuration. The "
    "independent review must check this data flow and preserve evaluation behavior. "
    "Mutate only train/model knobs, no evaluator, file path, input split, seed or metric changes. "
    "Keep a primary native metric suitable for comparing validation candidates. Do not use a training-set metric. "
    "If metric_policy is supplied, use its first target as the primary metric and preserve every declared target "
    "and hard-limit metric in the frozen scorer. Never rewrite user-approved targets or limits. "
    "When a structural study is feasible, name the exact mutable Python modules and the training/prediction stages "
    "that need them. Use stage_views to make those modules invisible to data preparation and frozen scoring, "
    "and give the fixed scorer only measured predictions or non-executable artifacts. Stage entrypoints themselves are frozen. "
    "If the existing scorer must import an editable module, leave the structural scope empty and explain the limitation. "
    "When training uses a validation file also read at the same path by the scorer, keep the training file fixed. "
    "Use score_input_bindings records with logical_path and source_path to map the scorer's original filename to a separate projected input visible "
    "only to the final scoring stage. Emit the validation and untouched test content for that alternate file in "
    "the respective views. Do not substitute the test file into a name earlier stages can read. "
    "For a project with original train/test, carve validation only from the original training set and reserve the "
    "original test samples for final evaluation. Prefer the complete untouched original test set when the frozen "
    "scorer supports it. If its native shape, batch count or denominator is fixed, freeze one compatible untouched "
    "window from the original held-out rows in publisher order before candidate dispatch. Exclude any rows already "
    "used in prior development when feasible. Declare actual original row IDs and the exact scope of this limited "
    "test in limitations. Do not change scorer code, fabricate rows, use padding or omit an executable test view. "
    "Re-run native feature fitting with the reduced training data if "
    "the existing features already used the validation rows for vocabulary, normalization, or learned transforms. "
    "The builder can run sealed original preprocessing scripts once to prepare both views. If the builder writes "
    "the original fit inputs, do not add preprocessing stages or repeat them each experiment. Only declare a new "
    "preparation stage if the projection emits an earlier input it genuinely consumes. If the measured baseline "
    "starts with preprocessing that the builder has already reproduced and materialized, list the replaced initial "
    "stage names in replaced_preparation and supply their outputs in projected_inputs. working_directories lists "
    "only those new stages, never the already measured train and score stages. For existing CV, preserve its "
    "actual folds and label it as exploration only. Never invent an independent test set. "
    "original_inputs are exact paths needed by the projection; include additional upstream artifacts if needed even "
    "when not currently listed. The controller requests missing files. Do not put such paths into blockers. "
    "Write a short split_builder Python script run at the isolated project root with original_inputs and audited "
    "source_files copied to their original relative paths. It may invoke the original preprocessing scripts in a "
    "temporary directory using sys.executable. It writes split.json and views/validation/<project-relative file> and, "
    "where independent test exists, views/test/<project-relative data file>. The stage runner receives only one view. "
    "It must write exactly those files. Both views must use exactly projected_inputs as their file lists. Use stable "
    "actual IDs from original rows. split.json is either {train,validation,test} lists of unique disjoint IDs, or "
    "{folds:[{id,train,validation}],heldout:{train,test}} if final training uses a different set of rows. Use null "
    "heldout if none exists. An original published test set may be locked out of this search and run once at the "
    "end as a historical benchmark. Disclose known historical use or unknown previous exposure in limitations. "
    "Only report blockers if no valid exploration can be run, rather than for limitations on claims. "
    "Use baseline_certificate and its stage receipts as evidence that the measured configuration ran. "
    "Do not treat absent user artifacts, a smaller accepted budget, or source excerpts as evidence against "
    "a successful measured configuration. An issue introduced by the new projection is a blocker only "
    "if it would change the measured behavior, violate data isolation, or prevent execution. "
    "Enter nonblocking scientific or reporting caveats under limitations or findings. "
    "A native scorer can receive validation data at its historical test path in the validation view. Use the "
    "frozen original held-out rows at that path in the test view. Do not access the network or execute tools."
)


def admission_sources(project, baseline, cards):
    ordered = dict.fromkeys([*(stage.script for stage in reversed(baseline.stages)), *baseline.source_files])
    texts, coverage = {}, {}
    remaining = ADMISSION_SOURCE_LIMIT
    for name in ordered:
        raw = resolve_inside(project, name).read_bytes()
        if digest(raw) != cards[name]["sha256"]:
            raise ValueError(f"Measured admission source changed: {name}")
        size = min(remaining, ADMISSION_FILE_LIMIT, len(raw))
        texts[name] = raw[:size].decode("utf-8", errors="replace")
        coverage[name] = {"complete": size == len(raw), "sha256": cards[name]["sha256"]}
        remaining -= size
    texts.update({name: card["excerpt"][:SUPPORT_EXCERPT] for name, card in cards.items() if name not in ordered})
    return texts, coverage


def baseline_receipt_summary(store, reference):
    content = store.get(reference)["content"]
    stages = []
    for key in content["stages"]:
        row = store.get(key)["content"]
        stages.append({"evidence": key, "stage": row["stage"], "status": row["status"], "files": row["files"]})
    return {"evidence": reference, "metrics": content["metrics"], "stages": stages}


def record_baseline_revalidation(store, admitted, measured):
    """Keep the admitted baseline when only the source manifest's ancestry changed."""
    store.verify_tree(admitted)

    def measurement(reference):
        certificate = store.get(reference)
        content = certificate["content"]
        if (certificate["kind"] != "measured_native_baseline"
                or certificate["parents"] != [content["recipe"], *content["stages"]]):
            raise ValueError("Expected a native baseline certificate with measured stage receipts")
        recipe = store.get(content["recipe"])
        if recipe["kind"] != "native_baseline_recipe" or len(recipe["parents"]) != 2:
            raise ValueError("Expected the approved baseline recipe with its source manifest and plan")
        sources, plan = recipe["parents"]
        source_manifest = store.get(sources)
        if source_manifest["kind"] != "onboarding_sources" or store.get(plan)["kind"] != "onboarding_plan":
            raise ValueError("Expected the approved onboarding manifest and plan")
        return canonical({
            "measurement": {name: value for name, value in content.items() if name != "recipe"},
            "recipe": recipe["content"],
            "sources": source_manifest["content"],
            "plan": plan,
        })

    if measurement(admitted) != measurement(measured):
        raise ValueError("Existing candidate admission has a different measured baseline")
    return store.put("native_baseline_revalidation", {"admitted": admitted, "measured": measured},
                     [admitted, measured])


def connect_native(root: Path, *, rounds: int, candidates: int, structural_files: int = 0,
                   metric_policy: MetricPolicy | None = None, worker=None) -> dict:
    folder = root / (f"native-structure-{structural_files}" if structural_files else "native-study")
    folder.mkdir(exist_ok=True)
    worker = worker or CodexWorker(root)
    with FileLock(str(folder / ".admission.lock"), timeout=0):
        return _connect(root, folder, rounds, candidates, worker, structural_files, metric_policy)


def _connect(root, folder, rounds, candidates, worker, structural_files=0, metric_policy=None):
    state_path = folder / "admission.json"
    original = json.loads((root / "native-onboarding.json").read_bytes())
    source = json.loads((root / "onboarding.json").read_bytes())
    store = Evidence(root)
    if original["phase"] != "baseline_verified":
        raise ValueError("Native candidate admission requires a measured baseline")
    store.verify_tree(original["evidence"])
    certificate = store.get(original["evidence"])
    if certificate["kind"] != "measured_native_baseline" or certificate["content"]["recipe"] != original["recipe"]:
        raise ValueError("Native baseline certificate does not match its recipe")
    record = json.loads(state_path.read_bytes()) if state_path.exists() else {}
    identity = {"rounds": rounds, "candidates": candidates}
    if metric_policy or record.get("metric_policy"):
        identity["metric_policy"] = metric_policy.model_dump() if metric_policy else None
    if record and any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Existing candidate admission has a different budget or metric policy")
    admitted = record["baseline"] if record else original["evidence"]
    identity["baseline"] = admitted
    if admitted != original["evidence"]:
        record["baseline_revalidation"] = record_baseline_revalidation(store, admitted, original["evidence"])
        original = {**original, "evidence": admitted}

    def save(phase, **values):
        record.update(identity, phase=phase, **values)
        atomic(state_path, canonical(record))
        return record

    study = root / (f"study-structure-{structural_files}" if structural_files else "study")
    if (study / "state.json").exists():
        if not record.get("revision") or Research(study).command.revision != record["revision"]:
            raise ValueError("Existing study does not belong to this native admission")
        return save("study_ready", study=str(study), issues=[])
    if metric_policy:
        missing = [metric.name for metric in metric_policy.measured_metrics()
                   if metric.name not in certificate["content"]["metrics"]]
        if missing:
            return save("awaiting_input", issues=[issue("metric_policy", "冻结的基线未测量指定指标。",
                ", ".join(missing))])
    project = Path(original.get("execution_project", source["project"]))
    interpreter = Path(original["interpreter"])
    artifacts = Path(original["artifacts"]) if original.get("artifacts") else None
    plan = store.get(original.get("execution_plan", source["plan"]))["content"]["plan"]
    baseline = NativeRecipe.model_validate(store.get(original["recipe"])["content"]["recipe"])
    cards = {card["path"]: card for card in store.get(original.get("execution_sources", source["sources"]))["content"]["cards"]}
    source_project = project
    refactor_proof = None
    structure_scope = None
    if structural_files and baseline.stage_views:
        stage_views = {item.stage: item for item in baseline.stage_views}
        protected = set(stage_views[baseline.stages[-1].name].source_files) | {stage.script for stage in baseline.stages}
        available = {name for stage in baseline.stages[:-1] for name in stage_views[stage.name].source_files}
        existing = [name for name in available if name.endswith(".py") and name not in protected]
        if len(existing) < structural_files:
            try:
                project, baseline, cards, refactor, refactor_proof = prepare_training_scope(
                    root, folder, project, interpreter, baseline, cards, original["evidence"], worker, structural_files)
            except (OSError, ValueError, WorkerFailure) as exc:
                return save("review", issues=[issue("training_scope", "结构训练边界未通过原生一致性核验。", str(exc))])
            plan = {**plan, "baseline_entrypoints": [stage.script for stage in baseline.stages[:-1]]}
            structure_scope = {"paths": [item.path for item in refactor.modules], "stages": [refactor.stage],
                               "qualification": refactor_proof,
                               "measurement": baseline_receipt_summary(store, refactor_proof)}
    try:
        excerpts, coverage = admission_sources(project, baseline, cards)
    except (OSError, ValueError) as exc:
        return save("integrity_error", issues=[issue("source_integrity", "候选准入的封存源码无法核对。", str(exc))])
    payload = {"objective": source["objective"], "plan": plan, "approved_baseline": baseline.model_dump(),
               "baseline_results": certificate["content"]["metrics"], "sources": excerpts,
               "source_coverage": coverage, "baseline_certificate": baseline_receipt_summary(store, original["evidence"]),
               "available_project_files": available_paths(source_project),
               "available_artifacts": available_paths(artifacts), "rounds": rounds,
               "required_structural_files": structural_files, "qualified_training_scope": structure_scope,
               "max_candidates": candidates, "requirements": PROJECTION_CONTRACT}
    if metric_policy:
        payload["metric_policy"] = metric_policy.model_dump()
    last = record.get("failure")
    contract = digest(canonical({"requirements": PROJECTION_CONTRACT, "schema": StudyDraft.model_json_schema()}))
    prior_blocker = None
    if record.get("draft") and record.get("phase") == "awaiting_input" and record.get("draft_contract") != contract:
        cached = store.get(record["draft"])["content"]["draft"]
        if cached["blockers"]:
            prior_blocker = record["draft"]
            payload["previous_protocol_blockers"] = cached["blockers"]
            save("draft", draft=None, draft_contract=contract)
    for _ in range(3):
        parents = [key for key in (original["evidence"], refactor_proof, last, prior_blocker) if key]
        try:
            draft_ref = record.get("draft")
            if draft_ref:
                draft = StudyDraft.model_validate(store.get(draft_ref)["content"]["draft"])
            else:
                if last:
                    payload["previous_failure"] = store.get(last)["content"]["reason"]
                draft, meta = worker.run("native-study-draft", payload, StudyDraft)
                draft_ref = store.put("native_study_draft", {"draft": draft.model_dump(), "worker": meta}, parents)
                save("draft", draft=draft_ref, draft_contract=contract)
            if draft.blockers:
                review, reviewer = worker.run("native-study-review", {
                    **payload, "draft": draft.model_dump(),
                    "requirements": PROJECTION_CONTRACT + " Accept these blockers only if they prevent valid local exploration. The "
                                    "published benchmark's prior test history limits claims about unseen cohorts. "
                                    "It does not preclude internal validation and a frozen historical test after "
                                    "candidate selection. Reject spurious blockers with the applicable scope.",
                }, NativeReview)
                blocker_ref = store.put("native_study_blocker_review", {"review": review.model_dump(),
                                          "worker": reviewer}, [draft_ref])
                parents.append(blocker_ref)
                if review.accepted:
                    return save("awaiting_input", draft_contract=contract,
                                issues=[issue("study_protocol", text) for text in draft.blockers])
                raise ValueError("Reclassify proposed blockers: " + " ".join(review.findings))
            if len(draft.mutable_sources) < structural_files:
                raise ValueError("The requested study must admit multiple training-only source files")
            if metric_policy and (draft.primary_metric, draft.direction) != (metric_policy.targets[0].name,
                                                                             metric_policy.targets[0].direction):
                raise ValueError("Study draft changed the first user-approved objective")
            if structure_scope and (set(draft.mutable_sources) != set(structure_scope["paths"])
                                    or draft.mutable_stages != structure_scope["stages"]):
                raise ValueError("The study must use the qualified training-only scope")
            recipe = recipe_for(project, baseline, draft, plan, cards)
            files, missing = required_inputs(source_project, artifacts, draft.original_inputs)
            if missing:
                return save("awaiting_input", issues=[issue("projection_inputs",
                    "请按原相对路径提供候选所需的原生数据，可放入已选 --artifacts 目录。", name)
                    for name in missing])
            input_hashes = {name: file_hash(path) for name, path in files.items()}
            review, reviewer = worker.run("native-study-review", {
                **payload, "draft": draft.model_dump(), "candidate_recipe": recipe.model_dump(),
                "verified_inputs": input_hashes,
                "requirements": "Independently audit the projection and axis bindings. " + PROJECTION_CONTRACT +
                                " candidate_recipe is the materialized recipe and inherits working directories "
                                "for existing train and scoring steps. The restriction to new steps applies only "
                                "to draft.working_directories. Use the qualified training-only scope when present. "
                                "Base the acceptance boolean on executable behavior and frozen protocol only. "
                                "Correct inaccurate nonblocking caveats in findings while accepting a sound recipe. "
                                " Reject held-out leakage, fabricated IDs, changes to the scorer, or scalar choices "
                                "that alter a data split, evaluation, model output routing or scoring semantics.",
            }, NativeReview)
            review_ref = store.put("native_study_review", {"review": review.model_dump(), "worker": reviewer}, [draft_ref])
            parents.append(review_ref)
            if not review.accepted:
                raise ValueError("Study admission rejected: " + " ".join(review.findings))
            repo = prepare_bundle(folder, interpreter, draft, recipe, project, files, cards, original["evidence"],
                                  input_hashes)
            revision = commit_bundle(repo)
            branches = min(3, len(draft.axes) + bool(draft.mutable_sources), max(1, candidates // rounds))
            config = CommandExperiment(repository=str(repo), revision=revision, command=["isolated-native"],
                split_manifest="split.json", baseline={axis.name: axis.baseline for axis in draft.axes},
                choices={axis.name: axis.choices for axis in draft.axes}, bridge_interpreter=str(interpreter),
                mutable_sources=draft.mutable_sources, min_structural_files=structural_files,
                timeout_seconds=sum(stage.timeout for stage in recipe.stages))
            metrics = metric_policy.measured_metrics() if metric_policy else [
                Metric(name=draft.primary_metric, direction=draft.direction,
                       threshold=certificate["content"]["metrics"][draft.primary_metric])]
            study_spec = Project(name=plan["project"], objective=source["objective"],
                knowledge="\n".join([plan["dataset_and_split"], draft.explanation,
                                    *original.get("limitations", []), *draft.limitations]),
                archive_uri=str(study), metrics=metrics, metric_policy=metric_policy,
                max_experiments=candidates, max_batches=rounds, branches=branches, concurrency=branches,
                reversible=True, affordable=True, confirmed_knowledge=True,
                independent_review=True, followup_experiments=True, command_experiment=config)
            checks = inspect(study_spec)
            if checks["issues"]:
                return save("awaiting_input", issues=checks["issues"], revision=revision, draft=draft_ref)
            save("initialize", revision=revision, review=review_ref, draft=draft_ref, issues=[])
            Research.initialize(study, study_spec)
            return save("study_ready", study=str(study), issues=[])
        except (OSError, ValueError, WorkerFailure, subprocess.SubprocessError) as exc:
            last = store.put("native_study_rejected", {"reason": str(exc)}, parents)
            save("draft", draft=None, failure=last)
    return save("awaiting_input", issues=[issue("study_protocol",
        "候选接入未通过，请查看候选资格审查和运行回执。", store.get(last)["content"]["reason"])])
