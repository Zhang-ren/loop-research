"""Discover and qualify native Python stages against a sealed project snapshot."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Literal

from filelock import FileLock
from pydantic import Field

from .models import StrictModel
from .onboard_replay import resolve_inside, runtime_interpreter
from .onboarding import SUPPORT_EXCERPT, onboard, source_cards, supporting_sources
from .preflight import issue
from .runtime_failure import classify_failure
from .sealed_stage import Stage, file_hash, stage_command
from .storage import Evidence, atomic, canonical, digest
from .workers import CodexWorker, WorkerFailure

MAX_ATTEMPTS = 3
MAX_METRIC_REPORT_BYTES = 4_000_000
MAX_NATIVE_STAGE_SECONDS = 3600


class FrozenInputChanged(ValueError):
    pass


class MetricPath(StrictModel):
    name: str = Field(min_length=1)
    keys: list[str] = Field(min_length=1, max_length=8)


class StageDirectory(StrictModel):
    stage: str
    path: str


class StageView(StrictModel):
    stage: str
    source_files: list[str]
    input_files: list[str]
    previous_outputs: list[str]


class ScoreInputBinding(StrictModel):
    logical_path: str
    source_path: str


class NativeRecipe(StrictModel):
    stages: list[Stage] = Field(max_length=6)
    source_files: list[str] = Field(max_length=24)
    seed_paths: list[str] = Field(max_length=12)
    attestations: list[str] = Field(max_length=8)
    readonly_directories: list[str] = Field(max_length=20)
    metrics_file: str | None
    metrics: list[MetricPath] = Field(max_length=16)
    questions: list[str] = Field(max_length=8)
    metrics_format: Literal["json", "key_value"] = "json"
    metrics_append: bool = False
    input_files: list[str] = Field(default_factory=list, max_length=16)
    working_directories: list[StageDirectory] = Field(default_factory=list, max_length=6)
    stage_views: list[StageView] = Field(default_factory=list, max_length=6)
    score_input_bindings: list[ScoreInputBinding] = Field(default_factory=list, max_length=16)

    def directories(self) -> dict[str, str]:
        return {item.stage: item.path for item in self.working_directories}

    def scorer_inputs(self) -> dict[str, str]:
        return {item.logical_path: item.source_path for item in self.score_input_bindings}


class NativeReview(StrictModel):
    accepted: bool
    findings: list[str]


def inside(path: str, ancestor: str) -> bool:
    return PurePosixPath(path).is_relative_to(PurePosixPath(ancestor))


def read_json(path: Path) -> dict:
    result = json.loads(path.read_bytes())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return result


def inventory(root: Path | None, focus: str | None) -> dict:
    if root is None:
        return {}
    source = resolve_inside(root, focus) if focus else root
    rows = {}
    for folder in (source, source / "output", source / "outputs"):
        if not folder.is_dir() or folder.is_symlink():
            continue
        rows[str(folder.relative_to(root))] = [
            item.name + ("/" if item.is_dir() else "") for item in
            sorted(folder.iterdir(), key=lambda x: x.name) if not item.name.startswith(".")
            and not item.is_symlink()
        ][:80]
    return rows


def required_inputs(project: Path, artifacts: Path | None, names: list[str]) -> tuple[dict[str, Path], list[str]]:
    found, missing = {}, []
    if len(set(names)) != len(names):
        raise ValueError("Required input files must be distinct")
    for name in names:
        if name != PurePosixPath(name).as_posix():
            raise ValueError(f"Required input file path must be project-relative: {name}")
        targets = [resolve_inside(root, name, existing=False) for root in (project, artifacts) if root is not None]
        source = next((path for path in targets if path.is_file()), None)
        if source is None:
            missing.append(name)
        else:
            found[name] = source
    return found, missing


def checked_stage_views(recipe: NativeRecipe) -> dict[str, StageView]:
    if not recipe.stage_views:
        if recipe.score_input_bindings:
            raise ValueError("Separate scoring inputs require explicit stage views")
        return {}
    global_access = [field for field in ("seed_paths", "attestations", "readonly_directories", "metrics_append")
                     if getattr(recipe, field)]
    if global_access:
        raise ValueError("Stage views require explicit files and fresh metric reports. Incompatible global fields: "
                         + ", ".join(global_access))
    views = {entry.stage: entry for entry in recipe.stage_views}
    if len(views) != len(recipe.stage_views) or views.keys() != {stage.name for stage in recipe.stages}:
        raise ValueError("Declare exactly one view for every stage")
    score = recipe.stages[-1].name
    bindings = recipe.scorer_inputs()
    ordinary_inputs = {name for stage, view in views.items() if stage != score for name in view.input_files}
    if (len(bindings) != len(recipe.score_input_bindings)
            or not bindings.keys() <= set(views[score].input_files)
            or not set(bindings.values()) <= set(recipe.input_files)
            or set(bindings.values()) & ordinary_inputs
            or any(name == target for name, target in bindings.items())):
        raise ValueError("A scoring input must use a separate committed file invisible to earlier stages")
    prior_outputs = set()
    for stage in recipe.stages:
        view = views[stage.name]
        inputs = {bindings.get(name, name) if stage.name == score else name for name in view.input_files}
        if (stage.script not in view.source_files or not set(view.source_files) <= set(recipe.source_files)
                or not inputs <= set(recipe.input_files)
                or not set(view.previous_outputs) <= prior_outputs):
            raise ValueError(f"Stage view contains undeclared files or earlier outputs: {stage.name}")
        selected = [*view.source_files, *view.input_files, *view.previous_outputs]
        if len(set(selected)) != len(selected):
            raise ValueError(f"Stage view has duplicate input paths: {stage.name}")
        if any(inside(name, other) or inside(other, name) for name in stage.outputs
               for other in prior_outputs):
            raise ValueError(f"Stage output overlaps an earlier stage output: {stage.name}")
        prior_outputs.update(stage.outputs)
    return views


def validate_recipe(recipe: NativeRecipe, project: Path, artifacts: Path | None, plan: dict, cards: dict) -> None:
    if recipe.questions:
        raise ValueError("Execution prerequisites still need an answer")
    if len(recipe.stages) < 2 or not recipe.metrics or not recipe.metrics_file:
        raise ValueError("Specify separate execution and scoring stages plus numeric metric paths")
    scripts = [step.script for step in recipe.stages]
    if (scripts[0] not in plan["baseline_entrypoints"] or scripts[-1] not in plan["independent_evaluation_entrypoints"]
            or not set(scripts) <= set(plan["baseline_entrypoints"] + plan["independent_evaluation_entrypoints"])):
        raise ValueError("Only reviewed executable entrypoints can be run, with independent scoring last")
    if len({s.name for s in recipe.stages}) != len(scripts) or len(set(scripts)) != len(scripts):
        raise ValueError("Use distinct stage names and entrypoints")
    if not (set(scripts) <= set(recipe.source_files) <= cards.keys()):
        raise ValueError("Every entrypoint must be a selected and sealed source file")
    outputs = Counter(path for step in recipe.stages for path in step.outputs)
    if (not any(inside(recipe.metrics_file, path) for path in recipe.stages[-1].outputs)
            or any(len(set(step.outputs)) != len(step.outputs) for step in recipe.stages)
            or any(count > (2 if recipe.metrics_append and path == recipe.metrics_file else 1)
                   for path, count in outputs.items())):
        raise ValueError("Declare distinct outputs except for an explicitly shared append-only metric report")
    directories = recipe.directories()
    if (len(directories) != len(recipe.working_directories)
            or not directories.keys() <= {s.name for s in recipe.stages}):
        raise ValueError("Working directory names must belong to declared stages")
    for path in directories.values():
        if path != "." and not resolve_inside(project, path).is_dir():
            raise ValueError(f"Invalid project-relative working directory: {path}")
    if len({metric.name for metric in recipe.metrics}) != len(recipe.metrics):
        raise ValueError("Metric names must be distinct")
    if recipe.metrics_append and recipe.metrics_format != "key_value":
        raise ValueError("Only native key-value metric reports may be shared with an earlier stage")
    if recipe.metrics_format == "key_value" and any(
        len(metric.keys) != 1 or not metric.keys[0].strip() for metric in recipe.metrics
    ):
        raise ValueError("Key-value reports require one exact metric label per metric")
    for step in recipe.stages:
        if step.gpu is not None or step.timeout > MAX_NATIVE_STAGE_SECONDS or any("\0" in arg for arg in step.arguments):
            raise ValueError("Native baseline uses a local CPU interpreter and a bounded timeout")
    sources = [*recipe.source_files, *recipe.attestations, *recipe.input_files, *recipe.scorer_inputs()]
    if any(inside(name, mount) for name in sources for mount in recipe.readonly_directories):
        raise ValueError("A read-only directory must not shadow a sealed source or attestation")
    if any(inside(name, other) or inside(other, name) for name in recipe.input_files
           for other in (*recipe.source_files, *recipe.attestations, *recipe.seed_paths)):
        raise ValueError("A required data file must be separate from sealed sources and prepared seed paths")
    for relative in outputs:
        resolve_inside(project, relative, existing=False)
        if any(inside(relative, item) or inside(item, relative) for item in sources + recipe.readonly_directories):
            raise ValueError(f"Output overlaps a sealed input: {relative}")
    for path in sources:
        resolve_inside(project, path, existing=False)
    if recipe.seed_paths and not artifacts:
        raise ValueError("Prepared artifacts root is missing")
    for relative in recipe.seed_paths:
        resolve_inside(artifacts, relative)
    for relative in recipe.readonly_directories:
        if not resolve_inside(project, relative).is_dir():
            raise ValueError(f"Read-only dependency is not a directory: {relative}")
    checked_stage_views(recipe)

def snapshot_files(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise FrozenInputChanged(f"Checkpoint contains an unsupported entry: {path.relative_to(root)}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = file_hash(path)
    return files


def output_hash(path: Path) -> str:
    if path.is_dir():
        outputs = snapshot_files(path)
        if not outputs:
            raise ValueError(f"Stage output directory is empty: {path.name}")
        return digest(canonical(outputs))
    return file_hash(path)


def stage_sources(project, artifacts, scratch, recipe, cards):
    scratch.mkdir(parents=True)
    for name in recipe.seed_paths:
        source = resolve_inside(artifacts, name)
        target = resolve_inside(scratch, name, existing=False)
        if source.is_dir():
            if any(p.is_symlink() for p in source.rglob("*")):
                raise ValueError(f"Artifact directory contains a symlink: {name}")
            shutil.copytree(source, target, dirs_exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            raise ValueError(f"Invalid prepared artifact: {name}")
    files, missing = required_inputs(project, artifacts, recipe.input_files)
    if missing:
        raise ValueError("Required data files are missing: " + ", ".join(missing))
    for name, source in files.items():
        target = resolve_inside(scratch, name, existing=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in recipe.source_files:
        raw = resolve_inside(project, name).read_bytes()
        if digest(raw) != cards[name]["sha256"]:
            raise FrozenInputChanged(f"Reviewed script or protocol changed: {name}")
        atomic(resolve_inside(scratch, name, existing=False), raw)
    for stage in recipe.stages:
        for name in stage.outputs:
            if resolve_inside(scratch, name, existing=False).exists():
                raise ValueError(f"Prepared artifacts already contain a declared new output: {name}")


def frozen_dependencies(project: Path, scratch: Path, recipe: NativeRecipe) -> dict[str, str]:
    hashes = {}
    def collect(path, expected):
        p = Path(path)
        if (not p.is_absolute() or ".." in p.parts or not isinstance(expected, str) or len(expected) != 64
                or any(value not in "0123456789abcdef" for value in expected)):
            raise ValueError("Native attestations require absolute file paths and SHA-256 hashes")
        if (not p.is_relative_to(project) and (any(part.startswith(".") for part in p.parts)
                or any(p.is_relative_to(base) for base in ("/proc", "/sys", "/dev", "/run", "/etc")))):
            raise ValueError("Native attestations cannot mount protected system paths")
        if hashes.setdefault(path, expected) != expected:
            raise ValueError(f"Conflicting attestations for {p.name}")

    for name in recipe.attestations:
        path = resolve_inside(scratch, name)
        value = read_json(path)
        for p, sha in value.get("source_hashes", {}).items():
            collect(p, sha)
        for relative, sha in value.get("files", {}).items():
            attested = resolve_inside(scratch, (Path(name).parent.parent / relative).as_posix(), existing=False)
            collect(str(project / attested.relative_to(scratch)), sha)
    return hashes


def bind_dependencies(command, project: Path, scratch: Path, dependencies, mounts):
    """Resolve checked source files against the original virtual project paths."""
    mark = command.index("--")
    options = []
    for name, expected in sorted(dependencies.items()):
        virtual = Path(name)
        physical = virtual
        if virtual.is_relative_to(project):
            relative = virtual.relative_to(project)
            local = resolve_inside(scratch, relative.as_posix(), existing=False)
            if not any(relative.is_relative_to(mount) for mount in map(PurePosixPath, mounts)):
                if local.exists() and local.is_file() and file_hash(local) == expected:
                    continue
                if local.exists() and (not local.is_file() or local.stat().st_size != 0):
                    raise FrozenInputChanged(f"Staged dependency differs from its attestation: {relative}")
                local.parent.mkdir(parents=True, exist_ok=True)
            else:
                if not physical.is_file() or file_hash(physical) != expected:
                    raise FrozenInputChanged(f"Read-only input changed or is missing: {relative}")
                continue
        if not physical.is_file() or file_hash(physical) != expected:
            raise FrozenInputChanged(f"Attested dependency changed or is missing: {virtual}")
        options.extend(("--ro-bind", str(physical.resolve(strict=True)), name))
    return [*command[:mark], *options, *command[mark:]]


def metric_report_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"Expected a native metric report file: {path.name}")
    with path.open("rb") as stream:
        content = stream.read(MAX_METRIC_REPORT_BYTES + 1)
    if len(content) > MAX_METRIC_REPORT_BYTES:
        raise ValueError(f"Native metric report exceeds the size limit: {path.name}")
    return content


def metric_at(value, keys: list[str]):
    for key in keys:
        if isinstance(value, dict):
            value = value[key]
        elif isinstance(value, list) and re.fullmatch(r"[0-9]{1,5}", key) and int(key) < len(value):
            value = value[int(key)]
        else:
            raise KeyError(key)
    return value


def read_metrics(final: Path, previous: Path, recipe: NativeRecipe) -> dict:
    path = resolve_inside(final, recipe.metrics_file)
    if recipe.metrics_format == "json":
        source = read_json(path)
    else:
        content = metric_report_bytes(path)
        prior = resolve_inside(previous, recipe.metrics_file, existing=False)
        if prior.exists():
            if not recipe.metrics_append:
                raise ValueError("The final score stage must create a new report or declare append-only scoring")
            prefix = metric_report_bytes(prior)
            if not content.startswith(prefix):
                raise FrozenInputChanged("Scoring rewrote earlier content in the append-only metric report")
            if len(content) == len(prefix) or (prefix and not prefix.endswith(b"\n")):
                raise ValueError("Scoring must append new metric records after a complete report line")
            content = content[len(prefix):]
        labels = {metric.keys[0] for metric in recipe.metrics}
        source = {}
        for line in content.decode("utf-8").splitlines():
            label, separator, value = line.partition(":")
            if separator and label.strip() in labels:
                label = label.strip()
                if label in source:
                    raise ValueError(f"Duplicate metric label in the new score output: {label}")
                try:
                    source[label] = float(value.strip())
                except ValueError as exc:
                    raise ValueError(f"Score output has a non-numeric metric: {label}") from exc

    values = {}
    for metric in recipe.metrics:
        try:
            value = metric_at(source, metric.keys)
        except KeyError as exc:
            raise ValueError(f"Frozen evaluator did not return metric {metric.name}") from exc
        if type(value) not in (int, float) or (type(value) is float and not isfinite(value)):
            raise ValueError(f"Frozen evaluator returned a non-numeric metric {metric.name}")
        values[metric.name] = value
    return values


def recipe_digest(recipe: NativeRecipe) -> str:
    # Keep paths to older JSON checkpoints stable.
    defaults = {"metrics_format", "metrics_append"} if recipe.metrics_format == "json" else set()
    if not recipe.input_files:
        defaults.add("input_files")
    if not recipe.working_directories:
        defaults.add("working_directories")
    if not recipe.stage_views:
        defaults.add("stage_views")
    if not recipe.score_input_bindings:
        defaults.add("score_input_bindings")
    return digest(canonical(recipe.model_dump(exclude=defaults)))


def materialize_stage_view(workspace: Path, seed: Path, outputs: dict[str, Path], view: StageView,
                           bindings: dict[str, str]) -> None:
    workspace.mkdir()
    sources = {name: resolve_inside(seed, name) for name in view.source_files}
    sources.update({name: resolve_inside(seed, bindings.get(name, name)) for name in view.input_files})
    sources.update({name: resolve_inside(outputs[name], name) for name in view.previous_outputs})
    for relative, source in sources.items():
        target = resolve_inside(workspace, relative, existing=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copyfile(source, target)


def run_recipe(project, root, interpreter, artifacts, recipe, cards, reuse=None, input_hashes=None,
               check_stage_output=None):
    """Each successful stage is an immutable checkpoint for its successor."""
    views = checked_stage_views(recipe)
    folder = root / "native-stages" / recipe_digest(recipe)
    folder.mkdir(parents=True, exist_ok=True)
    seed = folder / "input"
    stage_seed = folder / "input.json"
    if not stage_seed.exists():
        # An interrupted materialization never qualifies as a frozen checkpoint.
        seed = folder / f"input-{time.time_ns()}"
        stage_sources(project, artifacts, seed, recipe, cards)
        initial = {"workspace": str(seed), "dependencies": frozen_dependencies(project, seed, recipe),
                   "files": snapshot_files(seed)}
        if any(initial["files"].get(name) != sha for name, sha in (input_hashes or {}).items()):
            raise FrozenInputChanged("Required input changed after independent recipe review")
        atomic(stage_seed, canonical(initial))
    initial = read_json(stage_seed)
    seed = Path(initial["workspace"])
    dependencies = initial["dependencies"]
    if snapshot_files(seed) != initial["files"]:
        raise FrozenInputChanged("Frozen source materialization changed")
    expected_inputs = {**(input_hashes or {}), **{name: cards[name]["sha256"] for name in recipe.source_files}}
    if any(initial["files"].get(name) != sha for name, sha in expected_inputs.items()):
        raise FrozenInputChanged("Frozen source materialization does not match the reviewed files")
    previous = seed
    receipts = []
    values = None
    directories = recipe.directories()
    prior_checkpoints = {}
    for stage in recipe.stages:
        view = views.get(stage.name)
        final_stage = stage == recipe.stages[-1]
        bindings = recipe.scorer_inputs() if final_stage else {}
        report = ({"metrics": recipe.model_dump(include={"metrics", "metrics_file", "metrics_format", "metrics_append"})}
                  if final_stage and recipe.metrics_format != "json" else {})
        if stage.name in directories:
            report["working_directory"] = directories[stage.name]
        if view:
            report["stage_view"] = view.model_dump()
        if bindings:
            report["score_input_bindings"] = bindings
        signature = digest(canonical({"stage": stage.model_dump(), "sources": recipe.source_files,
                     "interpreter": str(interpreter), "dependencies": dependencies,
                     "mounts": recipe.readonly_directories, "seed_files": initial["files"],
                     "prior_outputs": [x["files"] for x in receipts], **report}))
        old = (reuse or {}).get(stage.name)
        pointer = folder / f"{stage.name}.json"
        cached = read_json(pointer) if pointer.exists() else old
        if not cached or cached["status"] != "passed" or cached["signature"] != signature:
            # Recover a pass recorded just before a process exit interrupted the pointer update.
            for path in sorted((folder / stage.name).glob("*/receipt.json"), reverse=True):
                candidate = read_json(path)
                if candidate["status"] == "passed" and candidate["signature"] == signature:
                    cached = candidate
                    break
        if cached and cached["status"] == "passed" and cached["signature"] == signature:
            checkpoint = Path(cached["workspace"])
            if snapshot_files(checkpoint) != cached["checkpoint"]:
                raise FrozenInputChanged(f"Completed stage checkpoint changed: {stage.name}")
            if check_stage_output:
                check_stage_output(stage, checkpoint)
            if final_stage:
                try:
                    values = read_metrics(checkpoint, previous, recipe)
                except FrozenInputChanged:
                    raise
                except (OSError, ValueError) as exc:
                    receipts.append({**cached, "status": "failed", "error": str(exc)})
                    return receipts, None
            receipts.append(cached)
            previous = checkpoint
            prior_checkpoints.update(dict.fromkeys(cached["files"], checkpoint))
            continue
        parent = folder / stage.name / str(time.time_ns())
        workspace = parent / "replica"
        if view:
            parent.mkdir(parents=True)
            materialize_stage_view(workspace, seed, prior_checkpoints, view, bindings)
        else:
            shutil.copytree(previous, workspace)
        visible_sources = ({name: initial["files"][bindings.get(name, name)]
                            for name in view.source_files + view.input_files} if view else initial["files"])
        visible_outputs = {name: sha for row in receipts for name, sha in row["files"].items()
                           if not view or name in view.previous_outputs}
        record = {"stage": stage.name, "signature": signature, "workspace": str(workspace),
                  "status": "interrupted", "started": time.time()}
        atomic(parent / "receipt.json", canonical(record))
        with (parent / "stdout.log").open("wb") as out, (parent / "stderr.log").open("wb") as err:
            try:
                command = stage_command(interpreter, workspace, project, stage.script, stage.arguments,
                                        mounts=recipe.readonly_directories,
                                        working_directory=directories.get(stage.name))
                command = bind_dependencies(command, project, workspace, dependencies, recipe.readonly_directories)
                result = subprocess.run(command, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                        check=False, timeout=stage.timeout)
                record["exit_code"] = result.returncode
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, stage.name)
                manifest = {}
                for name in [*stage.outputs, *visible_outputs]:
                    manifest[name] = output_hash(resolve_inside(workspace, name))
                shared_report = recipe.metrics_file if final_stage and recipe.metrics_append else None
                if any(manifest[name] != sha for name, sha in visible_outputs.items() if name != shared_report):
                    raise FrozenInputChanged("Stage modified an earlier completed output")
                checkpoint = snapshot_files(workspace)
                if any(checkpoint.get(name) != checksum for name, checksum in visible_sources.items()):
                    raise FrozenInputChanged("Stage modified a frozen source or prepared artifact")
                if check_stage_output:
                    check_stage_output(stage, workspace)
                if final_stage:
                    values = read_metrics(workspace, previous, recipe)
                record.update(status="passed", files={name: manifest[name] for name in stage.outputs},
                              checkpoint=checkpoint)
            except FrozenInputChanged:
                raise
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                failure_status, failure_category = classify_failure(exc)
                if isinstance(exc, FileNotFoundError) and record.get("exit_code") == 0:
                    failure_status, failure_category = "code_failed", "declared_output_missing"
                record.update(status="failed", error=str(exc), failure_status=failure_status,
                              failure_category=failure_category)
        record["finished"] = time.time()
        record["logs"] = str(parent)
        atomic(parent / "receipt.json", canonical(record))
        atomic(pointer, canonical(record))
        receipts.append(record)
        if record["status"] != "passed":
            return receipts, None
        previous = workspace
        prior_checkpoints.update(dict.fromkeys(record["files"], workspace))
    return receipts, values


def failure_context(receipts: list[dict], project: Path) -> dict:
    failure = receipts[-1]
    log = Path(failure["logs"]) / "stderr.log"
    # Keep runtime feedback useful without forwarding training data or full logs.
    missing = []
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size - 4096))
        tail = stream.read().decode(errors="replace")
    for value in re.findall(r"No such file or directory: ['\"]([^'\"]+)['\"]", tail):
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(project):
            path = path.relative_to(project)
        if not path.is_absolute() and ".." not in path.parts:
            missing.append({"path": str(path), "on_host": (project / path).exists()})
    last = tail.strip().splitlines()
    error = last[-1].partition(":")[0] if last else None
    if error and not re.fullmatch(r"[\w.]+", error):
        error = None
    module = re.search(r"ModuleNotFoundError: No module named ['\"]([\w.]+)['\"]", tail)
    locations = []
    for name, line in re.findall(r'(?m)^\s*File "([^"]+)", line ([0-9]+), in ', tail):
        path = Path(name)
        if (path.is_absolute() and path.is_relative_to(project) and path.suffix == ".py"
                and path.resolve() == path and path.is_file()):
            locations.append({"path": path.relative_to(project).as_posix(), "line": int(line)})
    label_type_mismatch = ("Labels in y_true and y_pred should be of the same type" in tail
                           or "Mix of label input types" in tail)
    details = {}
    if error in {"ValueError", "TypeError", "RuntimeError", "AssertionError", "KeyError"}:
        message = last[-1].partition(":")[2].strip().replace(str(project), "<project>")
        if message and re.fullmatch(r"[\w .,:;()\[\]'-]{1,240}", message):
            details["error_detail"] = message
    return {"stage": failure["stage"], "error": failure["error"], **details,
            "error_type": error, "missing_project_paths": missing,
            "missing_python_module": module.group(1) if module else None,
            "label_type_mismatch": label_type_mismatch, "source_locations": locations[-6:]}


def previous_failure_context(root: Path, project: Path, previous: dict) -> dict:
    """Recover newer sanitized diagnostics from an older failed stage receipt."""
    if "label_type_mismatch" in previous or not previous.get("stage") or not previous.get("previous_recipe"):
        return previous
    try:
        recipe = NativeRecipe.model_validate(previous["previous_recipe"])
        stage = next((item for item in recipe.stages if item.name == previous["stage"]), None)
        if stage is not None:
            receipt = read_json(root / "native-stages" / recipe_digest(recipe) / f"{stage.name}.json")
            if receipt.get("status") == "failed" and Path(receipt["logs"]).resolve().is_relative_to(root):
                return {**previous, **failure_context([receipt], project)}
    except (OSError, ValueError):
        # Diagnostics are optional when older logs have been archived.
        pass
    return previous


def qualify(project: Path, root: Path, objective: str, *, interpreter: Path | None = None,
            artifacts: Path | None = None, focus: str | None = None, worker=None, retry_adaptation=False) -> dict:
    project, root = project.resolve(strict=True), root.resolve()
    if root == project or project.is_relative_to(root):
        raise ValueError("Use a separate run directory")
    if artifacts:
        artifacts = artifacts.resolve(strict=True)
        if not artifacts.is_dir() or artifacts == root or artifacts.is_relative_to(root):
            raise ValueError("Prepared artifacts must be a separate existing directory")
    worker = worker or CodexWorker(root)
    source = onboard(project, root, objective, worker=worker, focus=focus,
                     available_artifacts=inventory(artifacts, focus))
    if source["phase"] != "plan_reviewed":
        return source
    with FileLock(str(root / ".native-onboarding.lock"), timeout=0):
        return _qualify(project, root, objective, source, interpreter, artifacts, worker, retry_adaptation=retry_adaptation)


def _qualify(project, root, objective, source, interpreter, artifacts, worker, *, retry_adaptation=False):
    store = Evidence(root)
    checkpoint = root / "native-onboarding.json"
    status = read_json(checkpoint) if checkpoint.exists() else {"phase": "recipe", "plan": source["plan"]}
    if status["plan"] != source["plan"]:
        raise ValueError("The run directory belongs to another reviewed project plan")

    def save(phase, **fields):
        status.update(phase=phase, **fields)
        atomic(checkpoint, canonical(status))
        return status

    if interpreter is None and status.get("interpreter"):
        interpreter = Path(status["interpreter"])
    if artifacts is None and status.get("artifacts"):
        artifacts = Path(status["artifacts"])
    if interpreter is None or not interpreter.is_file():
        return save("awaiting_input", issues=[issue("interpreter", "请提供项目已有 Python 解释器的绝对路径（--interpreter）。")])
    interpreter = runtime_interpreter(interpreter)
    if not os.access(interpreter, os.X_OK):
        return save("awaiting_input", issues=[issue("interpreter", "请提供可执行的项目 Python 解释器。")])
    identity = {"interpreter": str(interpreter), "artifacts": str(artifacts) if artifacts else None}
    if any(status.get(key) != value for key, value in identity.items()):
        if status.get("recipe"):
            raise ValueError("Use a new run directory to change the interpreter or prepared artifact root")
        save("recipe", **identity, attempts=0)
    plan = store.get(source["plan"])["content"]["plan"]
    entrypoints = set(plan["baseline_entrypoints"] + plan["independent_evaluation_entrypoints"])
    if not plan["baseline_entrypoints"]:
        return save("awaiting_input", **identity, issues=[issue("entrypoints",
             "请指出原项目可执行的训练入口。", "已审阅计划未能确认训练入口")])
    if not plan["independent_evaluation_entrypoints"]:
        from .native_adaptation import qualify_adaptation

        result = qualify_adaptation(project, root, objective, source, interpreter, artifacts, plan, worker,
                                    retry=retry_adaptation)
        return save(result.pop("phase"), **identity, **result)
    sources_id = status.get("execution_sources", source["sources"])
    material = store.get(sources_id)["content"]
    cards = {card["path"]: card for card in material["cards"]}
    if not status.get("execution_sources") and not status.get("recipe"):
        support = (supporting_sources(project, [*sorted(entrypoints), *cards], focus=source.get("focus"))
                   if "support_paths" not in material else [])
        if support:
            additions = source_cards(project, support, set(support), excerpt=SUPPORT_EXCERPT)
            cards.update({card["path"]: card for card in additions})
            sources_id = store.put("onboarding_sources", {"cards": list(cards.values())}, [sources_id])
        save("recipe", execution_sources=sources_id)
    excerpts = {name: card["excerpt"] for name, card in cards.items()}
    for name in (*sorted(entrypoints), *(item for item in cards if item.endswith("config.json"))):
        raw = resolve_inside(project, name).read_bytes()
        if digest(raw) != cards[name]["sha256"]:
            return save("integrity_error", issues=[issue("source_integrity", "接入证据中的源码已变化。", name)])
        excerpts[name] = raw.decode("utf-8")
    reusable = {}
    for ref in status.get("retained", []):
        item = store.get(ref)["content"]
        reusable[item["stage"]] = item
    while status.get("attempts", 0) < MAX_ATTEMPTS or status.get("recipe"):
        if not status.get("recipe"):
            payload = {"objective": objective, "plan": plan, "sources": excerpts, "python": str(interpreter),
                       "artifacts_available": artifacts is not None,
                       "artifact_inventory": inventory(artifacts, source.get("focus")),
                       "runtime_guarantees": [
                           "Each stage gets a fresh writable replica at the original absolute project path.",
                           "The Linux sandbox has no network; native loaders need local inputs or an existing offline mode.",
                           "An existing prepared artifact directory is copied, then checksummed before execution.",
                           "Declared outputs must be generated during this run. A key-value report may be shared "
                           "between training and scoring only when scoring appends to it without changing the prefix.",
                           "The runner verifies sealed source hashes and binds external attested files read-only "
                           "at their original absolute paths, including paths outside the project.",
                           "Passing stages and their output digests are reused during recovery.",
                           "The qualification scope ends after the original score command produces baseline metrics. "
                           "Separate report generation and full scientific verification have their own admission.",
                       ],
                       "requirements": "Return a minimal runnable baseline plan, only for the chosen current experiment. "
                           "Use the reviewed original Python train and independent score scripts with exact existing argv. "
                           "Stage script, source_files, readonly_directories, seed_paths, attestations and outputs are "
                           "project-root-relative, including any focus directory prefix. The last stage is the original "
                           "independent scorer. Use distinctive names train and score; target CPU only, gpu=null, "
                           "timeout <= 3600. Use the supplied imports and configs to derive commands, data paths, "
                           "and explicit working_directories entries with stage and path. Use '.' for commands documented from "
                           "the repository root and preserve a declared DVC wdir. Without this mapping the runner "
                           "uses the script's parent. An output directory may contain the final metric file. "
                           "split, dependencies and metrics. Do not ask for repository information supplied in sources. "
                           "Copy all needed scripts and protocol/config/sibling sources from sources. Use exact relative "
                           "paths. Verify command flags against every parser that processes them, and confirm that "
                           "stored parameters exist for the selected task and model. If a draft invocation is invalid, "
                           "use an equivalent supported invocation for the same experiment. For example, a native "
                           "search runner can launch one trial of a model with no tunable parameters. Do not require "
                           "changes to backend flags the chosen model does not use. Keep the user's baseline protocol "
                           "and any explicitly requested settings. "
                           "The prepared artifact root is a mirror of project paths. Declare required project-relative "
                           "individual data files in input_files whether or not present. The runner checks the original "
                           "project and artifact root and asks for exact missing paths; do not put them in questions. "
                           "Do not download data during training. "
                           "If it contains the original complete feature checkpoint, use seed_paths to copy that existing "
                           "experiment folder including audit support scripts and sealed outputs, excluding training and "
                           "scoring outputs which must be newly generated. List its JSON reports with source_hashes "
                           "or files as attestations. Name existing dependency directories that must mount read-only "
                           "using accurate code paths. Do not mount a directory over any copied source or input. "
                           "Distinguish prerequisites from training outputs. "
                           "Set metrics_format to json for nested numeric keys in a JSON object, or key_value for exact "
                           "labels on native 'Label: number' report lines. Declare the existing scorer report, never "
                           "invent a substitute JSON report. If the scorer appends to a report also written by training, "
                           "declare it in both stages and set metrics_append=true; the runner verifies the prefix and "
                           "only measures lines the scorer added. List all other outputs distinctly; declare files or "
                           "subdirectories without enclosing the shared report inside an earlier directory output. "
                           "No model code changes or scope expansion. If a prerequisite is specifically missing, "
                           "return questions listing only the missing facts, otherwise an empty questions array."}
            parents = list(dict.fromkeys([source["plan"], source["review"], source["sources"], sources_id]))
            for field, ref in (("previous_failure", status.get("last_failure")),
                               ("previous_runtime_failure", status.get("last_runtime_failure"))):
                if ref and ref not in parents:
                    payload[field] = previous_failure_context(root, project, store.get(ref)["content"])
                    parents.append(ref)
            try:
                recipe, meta = worker.run("onboard-native-recipe", payload, NativeRecipe)
                review_paths = set(plan["evidence_paths"] + recipe.source_files)
                review_input = {**payload, "recipe": recipe.model_dump(), "sources": {
                    name: excerpt if name in review_paths else excerpt[:800] for name, excerpt in excerpts.items()
                }}
                if recipe.questions:
                    review, reviewer = worker.run("onboard-native-review", {
                        **review_input,
                        "requirements": "Review proposed questions before interrupting the user. Accept only if all "
                        "questions require resources or decisions beyond the supplied repository evidence and the "
                        "existing environment. Reject requests for available commands, flags, config, metrics or "
                        "supported equivalent invocations. Cite concrete source paths and corrections when rejecting. "
                        "Corrections must use the approved native script with existing arguments, without inline code "
                        "or generated wrappers. Respect the original objective. Do not execute tools.",
                    }, NativeReview)
                    question_review = store.put("native_prerequisite_review", {
                        "questions": recipe.questions, "review": review.model_dump(), "worker": reviewer,
                    }, parents)
                    if not review.accepted:
                        parents.append(question_review)
                        raise ValueError("Questions can be resolved from the project: " + " ".join(review.findings))
                    return save("awaiting_input", **identity, prerequisite_review=question_review, issues=[
                        issue("prerequisite", question) for question in recipe.questions])
                validate_recipe(recipe, project, artifacts, plan, cards)
                supplied, missing = required_inputs(project, artifacts, recipe.input_files)
                if missing:
                    return save("awaiting_input", **identity, issues=[issue(
                        "input_files", "请把所缺数据按下列项目相对路径放入项目或 --artifacts 目录。", name)
                        for name in missing])
                input_hashes = {name: file_hash(file) for name, file in supplied.items()}
                approval, reviewer = worker.run("onboard-native-review", {
                    **review_input,
                    "verified_input_files": {name: {"sha256": input_hashes[name], "bytes": file.stat().st_size}
                                             for name, file in supplied.items()},
                    "requirements": "Independently verify the native recipe, required inputs and exact commands from the source. "
                        "Check that independent scoring follows training, outputs are fresh or safely append-only, "
                        "the original scripts, working directories and data/protocol are preserved, and metric names "
                        "exist in the scorer. verified_input_files are confirmed by the controller and take precedence "
                        "over a shallow artifact directory list. They will be checked when staged. "
                        "The reviewed plan is a draft. Allow supported invocations with equivalent behavior. One "
                        "trial of a model with no tunable parameters does not perform parameter selection. "
                        "Use the stated actual runtime guarantees. The task is measuring the native baseline through its "
                        "original separate scorer and numeric output. A report or a later interpretation validator is "
                        "outside this stage unless the original scoring script itself cannot execute without it. "
                        "Reject material defects; report concrete fixes. Do not execute tools.",
                }, NativeReview)
                approval_id = store.put("native_recipe_review", {"review": approval.model_dump(), "worker": reviewer}, parents)
                if not approval.accepted:
                    raise ValueError("Recipe review rejected: " + " ".join(approval.findings))
                key = store.put("native_baseline_recipe", {"recipe": recipe.model_dump(), "worker": meta,
                                                           "input_sha256": input_hashes},
                                [*parents, approval_id])
                save("execute", **identity, recipe=key, issues=[])
            except (OSError, ValueError, WorkerFailure) as exc:
                failure = store.put("native_recipe_rejected", {"reason": str(exc)}, parents)
                save("recipe", **identity, last_failure=failure, attempts=status.get("attempts", 0) + 1)
                continue

        key = status["recipe"]
        record = store.get(key)["content"]
        recipe = NativeRecipe.model_validate(record["recipe"])
        try:
            records, metrics = run_recipe(project, root, interpreter, artifacts, recipe, cards, reuse=reusable,
                                          input_hashes=record.get("input_sha256"))
            for receipt in records:
                if receipt["status"] == "passed":
                    reusable[receipt["stage"]] = receipt
            passed = [row for row in records if row["status"] == "passed"]
            retained = [store.put("native_stage_receipt", receipt, [key])
                        for receipt in (passed if metrics is not None else reusable.values())]
            if metrics is not None:
                certificate = {"recipe": key, "metrics": metrics, "stages": retained}
                receipt_id = store.put("measured_native_baseline", certificate, [key, *retained])
                return save("baseline_verified", **identity, metrics=metrics, evidence=receipt_id,
                            retained=retained, issues=[])
            reason = failure_context(records, project)
        except FrozenInputChanged as exc:
            return save("integrity_error", issues=[issue("source_integrity", "请核对已封存输入和已完成阶段的快照。", str(exc))])
        except (OSError, ValueError) as exc:
            retained = status.get("retained", [])
            reason = {"reason": str(exc)}
        failure = store.put("native_baseline_failed", {**reason, "previous_recipe": recipe.model_dump()}, [key, *retained])
        save("recipe", recipe=None, last_failure=failure, last_runtime_failure=failure,
             retained=retained, attempts=status.get("attempts", 0) + 1)
    return save("awaiting_input", issues=[issue("native_recipe",
        "请核对接入失败回执中的脚本、依赖或评分产物。", str(store.get(status["last_failure"])["content"]))])
