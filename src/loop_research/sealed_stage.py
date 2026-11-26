"""Run project preparation stages in a writable replica with read-only inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from filelock import FileLock
from pydantic import Field, model_validator

from .models import StrictModel
from .onboard_replay import ReplayRecipe, resolve_inside, sandbox
from .storage import atomic, canonical, digest


class Stage(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    script: str
    arguments: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(min_length=1)
    gpu: int | None = None
    timeout: int = Field(default=3600, ge=1, le=86400)


class StagePlan(StrictModel):
    project: Path
    scratch: Path
    interpreter: Path
    mounts: list[str] = Field(default_factory=list)
    external_inputs: list[Path] = Field(default_factory=list)
    files: dict[str, str]
    batches: list[list[Stage]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batches(self):
        names, outputs = set(), set()
        for batch in self.batches:
            if not batch:
                raise ValueError("Batches must contain at least one stage")
            gpus = set()
            for step in batch:
                if step.name in names or outputs.intersection(step.outputs) or set(step.outputs).intersection(self.files):
                    raise ValueError("Stages require distinct names and outputs separate from sealed inputs")
                if step.script not in self.files or len(set(step.outputs)) != len(step.outputs):
                    raise ValueError("Stage entrypoints must be sealed and their outputs unique")
                if step.gpu is not None:
                    if step.gpu in gpus:
                        raise ValueError("Parallel stages must use distinct GPUs")
                    gpus.add(step.gpu)
                names.add(step.name)
                outputs.update(step.outputs)
        return self


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_command(interpreter: Path, scratch: Path, project: Path, script: str, arguments: list[str], *,
                  mounts=(), external_inputs=(), readonly_files=None, gpu: int | None = None,
                  working_directory: str | None = None) -> list[str]:
    entrypoint = resolve_inside(scratch, script)
    directory = Path(script).parent.as_posix() if working_directory is None else working_directory
    if directory != "." and not resolve_inside(scratch, directory).is_dir():
        raise ValueError("Stage working directory must be inside the replica")
    recipe = ReplayRecipe(working_directory=directory, copy_files=[],
                          read_only_directories=list(mounts), seed_directories=[], steps=[],
                          evidence_files=[], prerequisites=[])
    command = sandbox(interpreter, scratch, project, recipe)
    marker = command.index("--")
    options = []
    for path in external_inputs:
        if path.parent != Path("/tmp") or path.resolve(strict=True) != path or not path.is_dir():
            raise ValueError("An external stage cache must be a specific regular directory under /tmp")
        options.extend(["--ro-bind", str(path), str(path)])
    for relative, path in (readonly_files or {}).items():
        target = resolve_inside(scratch, relative, existing=False)
        if target.exists() or path.resolve(strict=True) != path or not path.is_file():
            raise ValueError(f"A read-only stage asset must bind a regular file to a fresh path: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        options.extend(["--ro-bind", str(path), str(project / relative)])
    if gpu is not None:
        if type(gpu) is not int or gpu < 0:
            raise ValueError("GPU index must be a nonnegative integer")
        options.extend(["--setenv", "CUDA_VISIBLE_DEVICES", str(gpu)])
        devices = sorted(node for node in Path("/dev").glob("nvidia[0-9]*") if node.name[6:].isdigit())
        if Path(f"/dev/nvidia{gpu}") not in devices:
            raise ValueError(f"GPU {gpu} is unavailable")
        # CUDA enumerates the visible device nodes before applying physical indices.
        for node in (Path("/dev/nvidiactl"), Path("/dev/nvidia-uvm"), *devices):
            if not node.is_char_device():
                raise ValueError(f"Required GPU device is missing: {node}")
            options.extend(["--dev-bind", str(node), str(node)])
    script_arg = entrypoint.name if working_directory is None else str(project / script)
    return [*command[:marker], *options, *command[marker:], script_arg, *arguments]


def run_stage(work: Path, interpreter: Path, scratch: Path, project: Path, script: str,
              arguments: list[str], *, outputs: list[str], mounts=(), external_inputs=(),
              gpu: int | None = None, timeout: int = 3600) -> dict:
    """Persist exit status and output digests. A completed stage reopens its own receipt."""
    work.mkdir(parents=True, exist_ok=True)
    path = work / "receipt.json"
    signature = {"script": script, "script_sha256": file_hash(resolve_inside(scratch, script)),
                 "arguments": arguments, "outputs": outputs, "mounts": list(mounts),
                 "external_inputs": [str(item) for item in external_inputs], "gpu": gpu,
                 "interpreter": str(interpreter), "scratch": str(scratch), "project": str(project)}
    with FileLock(str(work / "stage.lock"), timeout=0):
        def output_hashes():
            return {item: file_hash(resolve_inside(scratch, item)) for item in outputs}

        if path.exists():
            prior = json.loads(path.read_bytes())
            if prior["signature"] != signature:
                raise ValueError("Stage differs from its original invocation")
            if prior["status"] != "passed":
                return prior
            if prior["files"] != output_hashes():
                raise ValueError("A completed stage output has changed")
            return prior
        if any(resolve_inside(scratch, item, existing=False).exists() for item in outputs):
            raise ValueError("Output exists without a completed receipt; inspect before retrying the stage")
        command = stage_command(interpreter, scratch, project, script, arguments, mounts=mounts,
                                external_inputs=external_inputs, gpu=gpu)
        started = time.time()
        record = {"signature": signature, "started": started, "status": "interrupted"}
        atomic(path, canonical(record))
        try:
            with (work / "stdout.log").open("wb") as out, (work / "stderr.log").open("wb") as err:
                completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                           timeout=timeout, check=False)
            record.update(exit_code=completed.returncode, status="failed")
            if completed.returncode == 0:
                record.update(files=output_hashes(), status="passed")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            record.update(status="failed", error=str(exc))
        record["elapsed_seconds"] = time.time() - started
        atomic(path, canonical(record))
        return record


def run_plan(plan_path: Path, work: Path) -> dict:
    plan = StagePlan.model_validate_json(plan_path.read_bytes())
    scratch = plan.scratch.resolve(strict=True)
    project = plan.project.resolve(strict=True)
    interpreter = plan.interpreter.resolve(strict=True)
    if scratch == project or project.is_relative_to(scratch):
        raise ValueError("Stages require a separate writable project replica")
    work.mkdir(parents=True, exist_ok=True)
    identity = digest(canonical(plan.model_dump(mode="json")))
    with FileLock(str(work / "plan.lock"), timeout=0):
        plan_id = work / "plan.sha256"
        if plan_id.exists() and plan_id.read_text() != identity:
            raise ValueError("Cannot revise a started plan in place")
        atomic(plan_id, identity.encode())

        def check_sources():
            for relative, expected in plan.files.items():
                if file_hash(resolve_inside(scratch, relative)) != expected:
                    raise ValueError(f"Sealed stage input changed: {relative}")

        def execute(step):
            return run_stage(work / step.name, interpreter, scratch, project, step.script, step.arguments,
                             outputs=step.outputs, mounts=plan.mounts, external_inputs=plan.external_inputs,
                             gpu=step.gpu, timeout=step.timeout)

        receipts = {}
        for batch in plan.batches:
            check_sources()
            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                receipts.update(zip((s.name for s in batch), pool.map(execute, batch), strict=True))
            if any(receipt["status"] != "passed" for receipt in receipts.values()):
                break
        check_sources()
        passed = sum(len(batch) for batch in plan.batches) == len(receipts) and all(
            receipt["status"] == "passed" for receipt in receipts.values()
        )
        record = {"status": "passed" if passed else "incomplete", "plan_sha256": identity,
                  "stages": {name: receipt["status"] for name, receipt in receipts.items()}}
        atomic(work / "plan-receipt.json", canonical(record))
        return record
