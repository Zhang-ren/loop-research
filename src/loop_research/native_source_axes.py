"""Bind reviewed scalar values in a sealed copy of an existing Python entrypoint."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import StrictModel
from .onboard_native import checked_stage_views
from .onboard_replay import resolve_inside
from .storage import atomic


class NativeSourceBinding(StrictModel):
    path: str
    kind: Literal["assignment", "keyword"]
    target: str = Field(min_length=1, max_length=120)
    default_source: str | None = None
    default_callable: str | None = None


def validate_source_scope(recipe, axis, training_stages):
    binding, stage = axis.source_binding, axis.stage
    views = checked_stage_views(recipe)
    score = recipe.stages[-1]
    path = Path(binding.path)
    if (axis.config_file or axis.argument or not views or stage not in training_stages
            or path.suffix != ".py" or path.is_absolute() or ".." in path.parts or path.as_posix() != binding.path
            or binding.path == score.script
            or binding.path not in views[stage].source_files):
        raise ValueError("A Python binding needs a sealed training source in an explicit train view")
    readers = {name for name, view in views.items() if binding.path in view.source_files}
    if binding.kind == "assignment":
        if not binding.target.isidentifier() or binding.default_source or binding.default_callable:
            raise ValueError("A Python assignment binding must name one existing top-level assignment")
        if not readers <= training_stages | {score.name}:
            raise ValueError("A configurable model literal must stay outside data preparation")
        return [binding.target, *axis.key_path]
    if len(axis.key_path) != 1 or not axis.key_path[0].isidentifier() or readers != {stage}:
        raise ValueError("A Python call binding needs one keyword in a training-only source file")
    if bool(binding.default_source) != bool(binding.default_callable):
        raise ValueError("A Python call default needs both its source and callable names")
    if binding.default_source and (not binding.default_source.endswith(".py")
                                   or binding.default_source not in views[stage].source_files):
        raise ValueError("A Python call default must be defined in the same sealed training view")
    return [*axis.key_path]


def _one(nodes, message):
    if len(nodes) != 1:
        raise ValueError(message)
    return nodes[0]


def _assignment(tree, name, keys):
    assignments = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if len(targets) != 1:
            raise ValueError(f"Python parameter {name} cannot use a chained assignment")
        assignments.append(node.value)
    current = _one(assignments, f"Expected one existing assignment to {name}")
    for key in keys:
        if not isinstance(current, ast.Dict) or any(not isinstance(item, ast.Constant) for item in current.keys):
            raise ValueError(f"Expected a literal dictionary in the path to {key}")
        current = _one([value for item, value in zip(current.keys, current.values, strict=True) if item.value == key],
                       f"Expected one existing literal dictionary key {key}")
    return current


def _call_default(project, binding, key):
    if not binding.default_source or not binding.default_callable:
        raise ValueError(f"No existing value or sealed default for Python keyword {key}")
    path = resolve_inside(project, binding.default_source)
    body = ast.parse(path.read_bytes(), filename=str(path)).body
    names = binding.default_callable.split(".")
    if len(names) == 2:
        owner = _one([node for node in body if isinstance(node, ast.ClassDef) and node.name == names[0]],
                     "Expected one sealed class defining the Python keyword default")
        body = owner.body
    elif len(names) != 1:
        raise ValueError("The default must name an existing function or a class method")
    function = _one([node for node in body if isinstance(node, ast.FunctionDef) and node.name == names[-1]],
                    "Expected one sealed function defining the Python keyword default")
    arguments = function.args
    positional = arguments.posonlyargs + arguments.args
    first_default = len(positional) - len(arguments.defaults)
    defaults = list(zip(positional[first_default:], arguments.defaults, strict=True))
    defaults.extend(zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True))
    return _one([value for arg, value in defaults
                 if arg not in arguments.posonlyargs and arg.arg == key and value is not None],
                f"The sealed function lacks a literal default for {key}")


def _call(tree, target):
    expression = ast.parse(target, mode="eval").body
    # Select existing syntax without interpreting callable expressions.
    if any(not isinstance(node, (ast.Name, ast.Attribute, ast.Subscript, ast.Constant, ast.Load))
           for node in ast.walk(expression)):
        raise ValueError("A Python call selector must name or look up the existing callable")
    selected = ast.dump(expression)
    return _one([node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.dump(node.func) == selected],
                f"Expected exactly one existing call to {target}")


def _offset(raw, line, column):
    return sum(len(item) + 1 for item in raw.split(b"\n", line - 1)[:line - 1]) + column


def apply_source_axis(project: Path, axis, value):
    if type(value) is not type(axis.baseline) or value not in axis.choices:
        raise ValueError(f"Python parameter outside its sealed choices: {axis.name}")
    binding = axis.source_binding
    path = resolve_inside(project, binding.path)
    raw = path.read_bytes()
    tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    replacement = repr(value).encode()
    insertion = None
    if binding.kind == "assignment":
        original = _assignment(tree, binding.target, axis.key_path)
    else:
        call = _call(tree, binding.target)
        key = axis.key_path[0]
        if any(item.arg is None for item in call.keywords):
            raise ValueError("Python call bindings cannot override unpacked keyword arguments")
        existing = [item.value for item in call.keywords if item.arg == key]
        if existing:
            original = _one(existing, f"Repeated Python keyword {key}")
        else:
            original = _call_default(project, binding, key)
            arguments = [*call.args, *call.keywords]
            if arguments:
                last = max(arguments, key=lambda item: (item.end_lineno, item.end_col_offset))
                insertion = _offset(raw, last.end_lineno, last.end_col_offset)
                replacement = b", " + key.encode() + b"=" + replacement
            else:
                insertion = _offset(raw, call.end_lineno, call.end_col_offset) - 1
                replacement = key.encode() + b"=" + replacement
    baseline = ast.literal_eval(original)
    if type(baseline) is not type(axis.baseline) or baseline != axis.baseline:
        raise ValueError(f"Python parameter default differs from measured baseline: {axis.name}")
    if value == baseline:
        return
    if insertion is None:
        start = _offset(raw, original.lineno, original.col_offset)
        end = _offset(raw, original.end_lineno, original.end_col_offset)
    else:
        start = end = insertion
    updated = raw[:start] + replacement + raw[end:]
    ast.parse(updated.decode("utf-8"), filename=str(path))
    atomic(path, updated)
