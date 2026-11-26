"""Snapshot public runtime versions and source bytes without copying user config."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from filelock import FileLock

from .storage import Evidence, atomic, canonical, digest


NUMERICAL_PACKAGES = {"numpy", "scipy", "scikit-learn"}
RUNTIME_PROBE = """
import importlib.metadata
import json
import sys
packages = {}
for name in json.loads(sys.stdin.read()):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({"python": list(sys.version_info[:2]), "packages": packages}))
"""


def snapshot(root: Path) -> dict:
    package = Path(__file__).parent
    files = {}
    for path in sorted(package.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(package).as_posix()
        raw = path.read_bytes()
        files[relative] = digest(raw)
        atomic(root / "runtime-source" / relative, raw)
    try:
        version = subprocess.check_output(["codex", "--version"], timeout=10, stderr=subprocess.STDOUT).decode().strip()
    except (OSError, subprocess.SubprocessError):
        version = "unavailable"
    value = {"python": sys.version, "platform": platform.platform(), "codex": version,
             "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]},
             "source_sha256": files, "model_actual": "Not exposed by the verified JSONL contract; not inferred from account defaults."}
    atomic(root / "runtime-manifest.json", canonical(value))
    return value


def archived_runtime(root: Path) -> tuple[str, dict]:
    pointer = root / "runtime-ref.json"
    if not pointer.is_file():
        raise ValueError("This run has no archived runtime. Open it with its original installation.")
    reference = json.loads(pointer.read_bytes())["evidence"]
    protocol = json.loads((root / "protocol-ref.json").read_bytes())["protocol"]
    evidence = Evidence(root)
    record = evidence.get(reference)
    if record["kind"] != "runtime_snapshot" or record["parents"] != [protocol]:
        raise ValueError("The archived runtime does not belong to this study protocol")
    evidence.verify_tree(reference)
    return reference, record["content"]


def verify_runtime_files(directory: Path, files: dict[str, str]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Missing regular archived runtime directory: {directory}")
    if not {"__init__.py", "cli.py"} <= files.keys():
        raise ValueError("The archived runtime does not contain the controller entrypoint")
    for relative, checksum in files.items():
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or name.as_posix() != relative:
            raise ValueError(f"Invalid archived runtime path: {relative}")
        file = directory / relative
        if not file.is_file() or file.resolve() != file or digest(file.read_bytes()) != checksum:
            raise ValueError(f"Archived runtime file changed: {relative}")
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink() or (path.is_file() and relative not in files):
            raise ValueError(f"Unexpected archived runtime file: {relative}")


def prepare_archived_runtime(root: Path, reference: str, files: dict[str, str]) -> Path:
    source = root / "runtime-source"
    verify_runtime_files(source, files)
    cache = root / ".runtime-exec"
    cache.mkdir(exist_ok=True)
    target = cache / reference
    with FileLock(str(cache / ".materialize.lock")):
        if not target.exists():
            with TemporaryDirectory(prefix=".materialize-", dir=cache) as work:
                staging = Path(work) / reference
                package = staging / "loop_research"
                for relative in files:
                    path = package / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes((source / relative).read_bytes())
                verify_runtime_files(package, files)
                staging.rename(target)
        verify_runtime_files(target / "loop_research", files)
    return target


def verify_runtime_environment(python: Path, manifest: dict, env: dict) -> None:
    expected = {name: version for name, version in manifest["packages"].items() if name in NUMERICAL_PACKAGES}
    result = subprocess.run([str(python), "-P", "-c", RUNTIME_PROBE], input=canonical(sorted(expected)),
                            env=env, check=True, capture_output=True, timeout=15)
    actual = json.loads(result.stdout)
    expected_python = [int(part) for part in manifest["python"].split()[0].split(".")[:2]]
    if actual["python"] != expected_python:
        raise ValueError("Select a Python interpreter with the archived major and minor version using --python")
    changed = [f"{name} requires {version}, found {actual['packages'].get(name)}"
               for name, version in expected.items() if actual["packages"].get(name) != version]
    if changed:
        raise ValueError("Select the archived numerical environment using --python: " + ", ".join(changed))


def resume_archived(root: Path, *, python: Path, interval: float, max_ticks: int) -> int:
    root = root.resolve(strict=True)
    python = python.absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Select an executable Python interpreter using --python")
    reference, manifest = archived_runtime(root)
    env = {**os.environ, "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    verify_runtime_environment(python, manifest, env)
    cache = prepare_archived_runtime(root, reference, manifest["source_sha256"])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(cache), env.get("PYTHONPATH")]))
    command = [str(python), "-P", "-m", "loop_research.cli", "run", str(root),
               "--interval", str(interval), "--max-ticks", str(max_ticks)]
    return subprocess.run(command, env=env, cwd=cache, check=False).returncode
