"""Execute an admitted native pipeline against exactly one sealed data view."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from pydantic import Field

from .command_candidate import candidate_parts
from .command_experiment import split_view
from .models import CommandExperiment, ParameterValue, StrictModel
from .native_source_axes import NativeSourceBinding, apply_source_axis, validate_source_scope
from .onboard_native import FrozenInputChanged, NativeRecipe, checked_stage_views, failure_context, run_recipe
from .runtime_failure import TrainingFailure, failed_result
from .onboard_replay import resolve_inside, runtime_interpreter
from .sealed_stage import file_hash
from .storage import atomic, canonical, digest


class NativeAxis(StrictModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    config_file: str | None
    key_path: list[str] = Field(max_length=8)
    stage: str | None
    argument: str | None
    baseline: ParameterValue
    choices: list[ParameterValue] = Field(min_length=2, max_length=16)
    source_binding: NativeSourceBinding | None = None


def validate_mutable_sources(recipe: NativeRecipe, files: list[str], mutable_stages: list[str]) -> None:
    if not files:
        if mutable_stages:
            raise ValueError("Structural stages require admitted mutable source files")
        return
    views = checked_stage_views(recipe)
    if not views:
        raise ValueError("Native structure changes require isolated stage views")
    writable = set(files)
    if len(writable) != len(files) or any(Path(name).suffix != ".py" for name in files):
        raise ValueError("Native source changes must name distinct Python files")
    selected = set(mutable_stages)
    if (not selected or len(selected) != len(mutable_stages)
            or not selected <= views.keys() - {recipe.stages[-1].name}):
        raise ValueError("Structural stages must be distinct, known training or prediction stages")
    protected_stages = views.keys() - selected
    protected = {stage.script for stage in recipe.stages}
    protected.update(name for stage in protected_stages for name in views[stage].source_files)
    execution_sources = {name for stage, view in views.items() if stage not in protected_stages
                         for name in view.source_files}
    if not writable <= execution_sources or writable & protected:
        raise ValueError("Mutable native sources must be execution modules invisible to the scorer and data preparation")


def apply_axes(project: Path, recipe: NativeRecipe, axes: list[NativeAxis], values: dict,
               preparation_stages=()) -> NativeRecipe:
    """Verify the measured defaults and replace only the declared scalar values."""
    recipe = recipe.model_copy(deep=True)
    configs = {}
    train_stages = {item.name for item in recipe.stages[:-1]} - set(preparation_stages)
    for axis in axes:
        option = values[axis.name]
        if axis.source_binding:
            validate_source_scope(recipe, axis, train_stages)
            apply_source_axis(project, axis, option)
        elif axis.config_file:
            path = resolve_inside(project, axis.config_file)
            if path not in configs:
                raw = path.read_bytes()
                document = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
                configs[path] = document
            field = configs[path]
            for key in axis.key_path[:-1]:
                if not isinstance(field, dict) or key not in field:
                    raise ValueError(f"Missing native parameter path for {axis.name}")
                field = field[key]
            name = axis.key_path[-1]
            if (not isinstance(field, dict) or name not in field or type(field[name]) is not type(axis.baseline)
                    or field[name] != axis.baseline):
                raise ValueError(f"Native parameter default differs from baseline: {axis.name}")
            field[name] = option
        else:
            step = next(item for item in recipe.stages if item.name == axis.stage)
            if step.arguments.count(axis.argument) != 1:
                raise ValueError(f"Native training option is not an explicit scalar: {axis.name}")
            index = step.arguments.index(axis.argument) + 1
            if index == len(step.arguments) or step.arguments[index] != str(axis.baseline):
                raise ValueError(f"Native training argument differs from baseline: {axis.name}")
            step.arguments[index] = str(option)
    for path, document in configs.items():
        atomic(path, canonical(document) if path.suffix == ".json"
               else yaml.safe_dump(document, sort_keys=False).encode())
    return recipe


def measure_native(root: Path, config: CommandExperiment, source: dict, code: str, seed: int,
                   split: str, logs: Path) -> dict:
    parameters, replacement = candidate_parts(code, config)
    _, train_ids, evaluation_ids = split_view(source["splits"], split)
    snapshot = root / "source.tar"
    if file_hash(snapshot) != source["snapshot_sha256"]:
        raise ValueError("Native experiment snapshot changed")
    logs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="native-input-", dir=logs) as temporary:
        directory = Path(temporary)
        project = directory / "native"
        data = directory / "data"
        project.mkdir()
        data.mkdir()
        prefix = f"views/{split}/"
        with tarfile.open(snapshot) as archive:
            plan = json.load(archive.extractfile("native-study.json"))
            for member in archive:
                if member.isdir():
                    continue
                if member.name.startswith("native/"):
                    base, name = project, member.name.removeprefix("native/")
                elif member.name.startswith(prefix):
                    base, name = data, member.name.removeprefix(prefix)
                else:
                    continue
                if not member.isfile():
                    raise ValueError("Native bundle inputs must be regular files")
                target = resolve_inside(base, name, existing=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as handle, target.open("xb") as output:
                    shutil.copyfileobj(handle, output)
        recipe = NativeRecipe.model_validate(plan["recipe"])
        if config.mutable_sources != plan.get("mutable_sources", []):
            raise ValueError("Native mutable source paths differ from the committed protocol")
        validate_mutable_sources(recipe, config.mutable_sources, plan.get("mutable_stages", []))
        for name, contents in (replacement or {}).items():
            atomic(resolve_inside(project, name), contents.encode())
        axes = [NativeAxis.model_validate(item) for item in plan["axes"]]
        recipe = apply_axes(project, recipe, axes, parameters, plan.get("preparation_stages", []))
        cards = {name: {"sha256": file_hash(resolve_inside(project, name))} for name in recipe.source_files}
        bound_sources = {axis.source_binding.path: cards[axis.source_binding.path]["sha256"]
                         for axis in axes if axis.source_binding}
        interpreter = runtime_interpreter(Path(config.bridge_interpreter))
        try:
            records, metrics = run_recipe(project, logs, interpreter, data, recipe, cards,
                                          input_hashes=plan["view_sha256"][split])
        except FrozenInputChanged as exc:
            raise TrainingFailure(str(exc), "unclassified_failed", "sealed_input_integrity") from exc
        if metrics is None:
            error = failure_context(records, project)
            failed = records[-1]
            raise TrainingFailure("Native stage failed: " + canonical(error).decode(),
                                  failed.get("failure_status", "unclassified_failed"),
                                  failed.get("failure_category", "stage_failure"))
        output = {"metrics": metrics}
        atomic(logs / "metrics.json", canonical(output))
    return {**output, "split": split, "seed": seed, "code_sha256": digest(code.encode()),
            "data_sha256": digest(canonical([source["snapshot_sha256"], source["splits"]])),
            "evaluation_indices": evaluation_ids, "train_indices": train_ids,
            "source_revision": source["revision"], "logs": str(logs),
            **({"bound_source_sha256": bound_sources} if bound_sources else {}),
            "native_stages": [{"stage": row["stage"], "files": row["files"],
                               "logs": row["logs"]} for row in records]}


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory / "request.json").read_bytes())
    key = digest(canonical(request))
    try:
        result = measure_native(directory.parent.parent, CommandExperiment.model_validate(request["command_experiment"]),
                                request["source"], request["code"], request["seed"], request["split"], directory)
        result["status"] = "succeeded"
    except Exception as exc:
        result = failed_result(exc, traceback.format_exc())
    atomic(directory / "result.json", canonical({**result, "request_sha256": key}))


if __name__ == "__main__":
    main()
