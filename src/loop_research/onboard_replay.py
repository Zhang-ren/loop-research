"""Execute a worker-proposed historical replay in an isolated Linux workspace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path, PurePosixPath

from filelock import FileLock
from pydantic import Field

from .models import StrictModel
from .storage import Evidence, atomic, canonical, digest
from .workers import CodexWorker, codex_executable


class ReplayStep(StrictModel):
    script: str
    arguments: list[str] = Field(max_length=8)


class ReplayRecipe(StrictModel):
    working_directory: str
    copy_files: list[str] = Field(max_length=35)
    read_only_directories: list[str] = Field(max_length=14)
    seed_directories: list[str] = Field(default_factory=list, max_length=3)
    steps: list[ReplayStep] = Field(max_length=6)
    evidence_files: list[str] = Field(max_length=8)
    prerequisites: list[str]


def resolve_inside(root: Path, relative: str, *, existing=True) -> Path:
    path = PurePosixPath(relative)
    if not path.parts or path.is_absolute() or ".." in path.parts or any(part.startswith(".") for part in path.parts):
        raise ValueError(f"Expected a visible relative path inside the project: {relative}")
    target = root.joinpath(*path.parts)
    if target.resolve(strict=existing) != target:
        raise ValueError(f"Symlink paths are not valid replay inputs: {relative}")
    return target


def runtime_interpreter(path: Path) -> Path:
    """Keep the entry path of a venv even when its executable is a symlink."""
    entry = path.parent.resolve(strict=True) / path.name
    resolved = entry.resolve(strict=True)
    return entry if (entry.parent.parent / "pyvenv.cfg").is_file() else resolved


def runtime_mounts(interpreter: Path) -> list[Path]:
    prefix = interpreter.parent.parent
    mounts = [prefix] if not prefix.is_relative_to("/usr") else []
    config = prefix / "pyvenv.cfg"
    if not config.is_file():
        return mounts
    entries = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        key, marker, value = line.partition("=")
        if marker:
            entries[key.strip()] = value.strip()
    home = Path(entries.get("home", ""))
    if not home.is_absolute() or home.name != "bin" or ".." in home.parts:
        raise ValueError("A virtual environment must name its absolute base Python bin directory")
    if home.is_relative_to("/usr") or home == Path("/bin"):
        return mounts
    version = re.match(r"(\d+\.\d+)(?:\.|$)", entries.get("version", entries.get("version_info", "")))
    if not version:
        raise ValueError("The virtual environment does not declare a supported Python version")
    libraries = [home.parent / name / f"python{version[1]}" for name in ("lib", "lib64")]
    stdlib = next((path for path in libraries if (path / "encodings" / "__init__.py").is_file()), None)
    if stdlib is None:
        raise ValueError("The base virtual-environment Python standard library is missing")
    executable = interpreter.resolve(strict=True)
    if not executable.is_relative_to(prefix):
        if not executable.is_relative_to(home.resolve(strict=True)):
            raise ValueError("The virtual-environment executable must belong to its declared base Python")
        mounts.append(home)
    # Compiled standard-library modules load sibling libraries such as libffi.
    return [*mounts, stdlib.parent]


def sandbox(interpreter: Path, scratch: Path, project: Path, recipe: ReplayRecipe,
            sealed_inputs: dict[str, str] | None = None) -> list[str]:
    if not sys.platform.startswith("linux"):
        raise ValueError("Historical replay isolation currently requires Linux")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        bundled = Path(codex_executable()).parent.parent / "codex-resources" / "bwrap"
        if bundled.is_file() and os.access(bundled, os.X_OK):
            bwrap = str(bundled)
    if not bwrap:
        raise ValueError("No bubblewrap executable available for isolated replay")
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
               "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "--setenv", "OPENBLAS_NUM_THREADS", "1",
               "--setenv", "OMP_NUM_THREADS", "1"]
    for path in ("/usr", "/bin", "/lib", "/lib64", "/etc/ld.so.cache"):
        if Path(path).exists():
            command.extend(["--ro-bind", path, path])
    for path in runtime_mounts(interpreter):
        command.extend(["--ro-bind", str(path), str(path)])
    # Preserve paths embedded in protocols and seals inside the isolated namespace.
    command.extend(["--bind", str(scratch), str(project)])
    for relative in recipe.read_only_directories:
        original = resolve_inside(project, relative)
        if not original.is_dir():
            raise ValueError(f"Read-only input is not a directory: {relative}")
        mount = resolve_inside(scratch, relative, existing=False)
        mount.mkdir(parents=True, exist_ok=True)
        command.extend(["--ro-bind", str(original), str(project / relative)])
    mounts = [PurePosixPath(item) for item in recipe.read_only_directories]
    for relative, expected in (sealed_inputs or {}).items():
        original = resolve_inside(project, relative)
        local = resolve_inside(scratch, relative, existing=False)
        with original.open("rb") as data:
            if hashlib.file_digest(data, "sha256").hexdigest() != expected:
                raise ValueError(f"Historical sealed input changed: {relative}")
        if local.is_file() or any(mount in PurePosixPath(relative).parents for mount in mounts):
            continue
        if relative in recipe.evidence_files:
            raise ValueError("A newly produced output cannot be bound from historical evidence")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.touch(exist_ok=False)
        command.extend(["--ro-bind", str(original), str(original)])
    command.extend(["--chdir", str(project / recipe.working_directory), "--", str(interpreter), "-B"])
    return command


def validate_recipe(project: Path, recipe: ReplayRecipe, admitted: set[str]):
    scripts = [script_path(recipe, step).as_posix() for step in recipe.steps]
    if len(scripts) < 2 or len(set(scripts)) < 2 or not recipe.evidence_files:
        raise ValueError("Provide separate historical execution and scoring steps with original JSON evidence")
    if not set(scripts) <= admitted or not set(scripts) <= set(recipe.copy_files):
        raise ValueError("Every executed script must appear in the audited evidence and copied files")
    controlled = list(map(PurePosixPath, recipe.read_only_directories))
    if any(path == PurePosixPath(file) or path in PurePosixPath(file).parents for path in controlled
           for file in [*recipe.copy_files, *recipe.evidence_files, *recipe.seed_directories]):
        raise ValueError("Inputs and outputs may not overlap a read-only directory")
    for file in recipe.copy_files:
        original = resolve_inside(project, file)
        if original.suffix.casefold() not in {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}:
            raise ValueError(f"Only existing project source or protocol text may be copied: {file}")
        if not original.is_file() or original.stat().st_size > 256_000:
            raise ValueError(f"Replay source is missing or too large: {file}")
    for step in recipe.steps:
        if any(not arg.startswith("--") or "=" in arg for arg in step.arguments):
            raise ValueError("Historical replay uses standalone long flags from the existing entrypoint")
    for relative in recipe.evidence_files:
        resolve_inside(project, relative, existing=False)
        if not relative.endswith(".json"):
            raise ValueError("Preserve original JSON evidence files")


def script_path(recipe: ReplayRecipe, step: ReplayStep) -> PurePosixPath:
    path = PurePosixPath(step.script)
    directory = PurePosixPath(recipe.working_directory)
    return path if path.is_relative_to(directory) else directory / path


def seed_historical_inputs(project, scratch, recipe):
    total = 0
    files = {}
    for relative in recipe.seed_directories:
        source = resolve_inside(project, relative)
        working = project if recipe.working_directory == "." else resolve_inside(project, recipe.working_directory)
        if not source.is_dir() or not source.is_relative_to(working):
            raise ValueError("Seed directories must belong to the selected experiment")
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Symlink found in the historical input directory")
            if not path.is_file():
                continue
            name = path.relative_to(project).as_posix()
            if name in recipe.evidence_files:
                continue
            total += path.stat().st_size
            if total > 384_000_000 or len(files) >= 1500:
                raise ValueError("Historical seeded files exceed the replay budget; use read-only inputs")
            target = resolve_inside(scratch, name, existing=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            files[name] = digest(target.read_bytes())
    return files


def historical_dependencies(project: Path, entrypoints: set[str]):
    """Read prior sealed input references without exposing sample names to a worker."""
    files = {}
    for location in sorted({(project / item).parent for item in entrypoints}):
        evidence = location / "output" / "summary.json"
        if not evidence.is_file() or evidence.is_symlink() or evidence.stat().st_size > 4_000_000:
            continue
        record = json.loads(evidence.read_bytes())
        for key in ("source_sha256", "input_hashes", "source_hashes", "input_sha256"):
            entries = record.get(key, {})
            if not isinstance(entries, dict):
                continue
            for name, expected in entries.items():
                path = Path(name)
                if (path.is_absolute() and path.is_relative_to(project) and isinstance(expected, str)
                        and len(expected) == 64 and all(char in "0123456789abcdef" for char in expected)):
                    name = path.relative_to(project).as_posix()
                    if files.setdefault(name, expected) != expected:
                        raise ValueError(f"Conflicting historical input hashes: {name}")
    return files


def propose_and_run(root: Path, interpreter: Path, worker=None, *, revise=False) -> dict:
    root = root.resolve(strict=True)
    interpreter = interpreter.resolve(strict=True)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("Provide the existing experiment Python interpreter")
    with FileLock(str(root / ".onboarding.lock"), timeout=0):
        return _propose_and_run(root, interpreter, worker or CodexWorker(root), revise)


def _propose_and_run(root, interpreter, worker, revise):
    state = json.loads((root / "onboarding.json").read_bytes())
    if state["phase"] != "plan_reviewed":
        raise ValueError("Review the onboarding plan first")
    project = Path(state["project"])
    evidence = Evidence(root)
    plan = evidence.get(state["plan"])["content"]["plan"]
    cards = evidence.get(state["sources"])["content"]["cards"]
    paths = {card["path"] for card in cards}
    entrypoints = set(plan["baseline_entrypoints"] + plan["independent_evaluation_entrypoints"])
    replay_cards = []
    for card in cards:
        if card["path"] not in entrypoints:
            continue
        data = resolve_inside(project, card["path"]).read_bytes()
        text = data.decode("utf-8")
        replay_cards.append({**card, "excerpt": text if len(text) <= 7_000 else text[:3_500] + "\n…\n" + text[-3_500:]})
    sealed_inputs = historical_dependencies(project, entrypoints)
    groups = sorted({Path(*Path(item).parts[:2]).as_posix() for item in sealed_inputs if len(Path(item).parts) >= 3})
    request = {
        "plan": plan, "sources": replay_cards, "sealed_input_directories": groups,
        "requirements": "Propose one small CPU historical baseline replay. Use existing scripts and options only. "
        "Provide a single working_directory, separate execution and scoring steps, and original output JSON paths. "
        "Copy only needed repository text files. Seed previously produced results from a small experiment directory "
        "if its established historical evaluator consumes them. The runner copies them to scratch and excludes named "
        "evidence_files from seeding so those must be produced during this execution. For existing read-only dependencies use relative directory paths "
        "outside the working directory, preserving repository layout. For a historical replay of existing samples, "
        "copy prior outputs only via seed_directories. Every executed script must be in the supplied source paths. "
        "Large feature caches, labels, checkpoints and prior sealed results may be mounted read-only when necessary. "
        "If these prerequisites cannot be justified, list them and return no steps. Do not execute any command.",
    }
    recipe_reference = root / "replay-recipe.json"
    reference = root / "replay-execution.json"
    previous = json.loads(reference.read_bytes()) if reference.exists() else None
    if recipe_reference.exists() and not revise:
        recipe_id = json.loads(recipe_reference.read_bytes())["evidence"]
        saved = evidence.get(recipe_id)
        if state["plan"] not in saved["parents"]:
            raise ValueError("Replay recipe belongs to another reviewed plan")
        recipe = ReplayRecipe.model_validate(saved["content"]["recipe"])
    else:
        parents = [state["plan"], state["review"], state["sources"]]
        if revise:
            if not previous or previous["phase"] != "replay_failed":
                raise ValueError("A replay can be revised only after a recorded execution failure")
            old = evidence.get(previous["evidence"])["content"]
            request["previous_recipe"] = evidence.get(old["recipe"])["content"]["recipe"]
            request["previous_failure"] = replay_failure_details(project, old)
            request["requirements"] += " Correct the previous recipe using the recorded execution failure. Preserve its selected baseline."
            parents.append(previous["evidence"])
        recipe, metadata = worker.run("onboard-replay", request, ReplayRecipe)
        recipe_id = evidence.put("onboarding_replay_recipe", {"recipe": recipe.model_dump(), "worker": metadata},
                                  parents)
        atomic(recipe_reference, canonical({"evidence": recipe_id, **recipe.model_dump()}))
    if not recipe.steps:
        return {"phase": "replay_blocked", "recipe": recipe_id, "prerequisites": recipe.prerequisites}
    validate_recipe(project, recipe, paths)
    for card in cards:
        path = resolve_inside(project, card["path"])
        if digest(path.read_bytes()) != card["sha256"]:
            raise ValueError(f"Reviewed source changed: {card['path']}")
    if previous and previous["recipe"] == recipe_id:
        record = evidence.get(previous["evidence"])["content"]
        if record["phase"] == "replay_executed" and record.get("interpreter") == str(interpreter):
            for file, expected in record["outputs"].items():
                if digest(resolve_inside(Path(record["workspace"]), file).read_bytes()) != expected:
                    raise ValueError(f"Previously produced replay evidence changed: {file}")
            return {**record, "evidence": previous["evidence"]}
    attempt_id = f"{recipe_id}-{uuid.uuid4().hex[:8]}"
    scratch = root / "historical-replay" / attempt_id
    scratch.mkdir(parents=True)
    copied = seed_historical_inputs(project, scratch, recipe)
    for file in recipe.copy_files:
        data = resolve_inside(project, file).read_bytes()
        atomic(resolve_inside(scratch, file, existing=False), data)
        copied[file] = digest(data)
    command = sandbox(interpreter, scratch, project, recipe, sealed_inputs)
    receipts = []
    output = root / "historical-replay" / (attempt_id + "-logs")
    output.mkdir()
    for index, step in enumerate(recipe.steps):
        args = [*command, str(project / script_path(recipe, step)), *step.arguments]
        with (output / f"{index}.stdout").open("wb") as stdout, (output / f"{index}.stderr").open("wb") as stderr:
            try:
                result = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=90, check=False)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                returncode = 124
        receipt = {"step": step.model_dump(), "returncode": returncode,
                   "stdout": str(output / f"{index}.stdout"), "stderr": str(output / f"{index}.stderr")}
        receipts.append(receipt)
        atomic(output / f"{index}.json", canonical(receipt))
        if returncode:
            break
    outputs = {}
    if len(receipts) == len(recipe.steps) and all(item["returncode"] == 0 for item in receipts):
        for file in recipe.evidence_files:
            path = resolve_inside(scratch, file, existing=False)
            if path.is_file() and path.stat().st_size <= 64_000_000:
                data = path.read_bytes()
                json.loads(data)
                outputs[file] = digest(data)
    success = len(outputs) == len(recipe.evidence_files)
    result = {"phase": "replay_executed" if success else "replay_failed", "recipe": recipe_id,
              "copied": copied, "outputs": outputs, "receipts": receipts, "workspace": str(scratch),
              "sealed_inputs": sealed_inputs, "interpreter": str(interpreter)}
    parents = [recipe_id] + ([previous["evidence"]] if previous and previous["recipe"] == recipe_id else [])
    key = evidence.put("onboarding_replay_execution", result, parents)
    atomic(reference, canonical({**result, "evidence": key}))
    return {**result, "evidence": key}


def replay_failure_details(project, record):
    failed = next((item for item in record["receipts"] if item["returncode"]), None)
    if not failed:
        return {"reason": "Named result files were missing after execution"}
    stderr = Path(failed["stderr"]).read_text(errors="replace")[-2400:]
    missing = []
    for match in re.findall(r"No such file or directory: ['\"]([^'\"]+)", stderr):
        path = Path(match)
        if path.is_absolute() and path.is_relative_to(project) and path.is_file():
            parts = path.relative_to(project).parts
            if len(parts) >= 3:
                missing.append(Path(*parts[:2]).as_posix())
    last = stderr.strip().splitlines()
    return {"step": failed["step"], "returncode": failed["returncode"],
            "missing_input_directories": sorted(set(missing)),
            "error_type": last[-1].partition(":")[0] if last else "unreported"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--revise", action="store_true")
    args = parser.parse_args()
    result = propose_and_run(args.root, args.interpreter, revise=args.revise)
    print(json.dumps({key: value for key, value in result.items() if key not in {"copied", "sealed_inputs"}},
                     ensure_ascii=False, indent=2))
    if result["phase"] != "replay_executed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
