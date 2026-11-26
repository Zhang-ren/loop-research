"""Candidate representation for parameter-only and isolated prediction experiments."""
from __future__ import annotations

import ast
import json
import tarfile
from pathlib import Path

from .command_experiment import checked_parameters
from .storage import canonical

STRUCTURE_TARGET = "model.structure"
PREDICTION_FILES = ("model.py", "predict.py", "score.py", "samples.json")
CODE_LIMIT = 48_000


def structural(config) -> bool:
    return config.command == ["isolated-predict"] or bool(config.mutable_sources)


def snapshot_file(archive: tarfile.TarFile, name: str, limit: int) -> bytes:
    member = archive.getmember(name)
    if not member.isfile() or not 0 < member.size <= limit:
        raise ValueError(f"Invalid regular snapshot file: {name}")
    with archive.extractfile(member) as stream:
        return stream.read()


def candidate_parts(code: str, config) -> tuple[dict, str | dict[str, str] | None]:
    if not structural(config):
        return checked_parameters(code, config), None
    value = json.loads(code)
    if not isinstance(value, dict) or set(value) != {"parameters", "model"}:
        raise ValueError("Isolated candidate needs parameters and model fields")
    parameters = checked_parameters(canonical(value["parameters"]).decode(), config)
    model = value["model"]
    if config.mutable_sources:
        if (not isinstance(model, dict) or model.keys() != set(config.mutable_sources)
                or any(not isinstance(text, str) or not text.strip() for text in model.values())):
            raise ValueError("Native candidate must contain only the exact admitted source files")
        if sum(len(text.encode()) for text in model.values()) > CODE_LIMIT:
            raise ValueError("The candidate source bundle exceeds its size limit")
        for name, text in model.items():
            try:
                ast.parse(text, filename=name)
            except SyntaxError as exc:
                raise ValueError(f"Invalid candidate Python in {name}: {exc.msg}") from exc
        return parameters, model
    if not isinstance(model, str) or not model.strip() or len(model.encode()) > CODE_LIMIT:
        raise ValueError("Candidate model must be a small Python module")
    try:
        node = ast.parse(model, filename="model.py")
    except SyntaxError as exc:
        raise ValueError(f"Invalid candidate Python: {exc.msg}") from exc
    if not any(isinstance(item, ast.FunctionDef) and item.name == "build_model" for item in node.body):
        raise ValueError("Candidate must define build_model(seed, parameters)")
    return parameters, model.rstrip() + "\n"


def candidate_code(parameters: dict, model: str | dict[str, str] | None) -> str:
    return canonical(parameters if model is None else {"parameters": parameters, "model": model}).decode()


def frozen_baseline_code(root: Path, config) -> str:
    if not structural(config):
        return candidate_code(config.baseline, None)
    with tarfile.open(root / "source.tar") as archive:
        model = ({name: snapshot_file(archive, f"native/{name}", CODE_LIMIT).decode()
                  for name in config.mutable_sources}
                 if config.mutable_sources else snapshot_file(archive, "model.py", CODE_LIMIT).decode())
    parameters, model = candidate_parts(candidate_code(config.baseline, model), config)
    return candidate_code(parameters, model)
