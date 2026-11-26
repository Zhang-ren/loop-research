"""Freeze a qualified bridge and launch a measured integration control."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from filelock import FileLock

from .command_experiment import prepare
from .models import CommandExperiment
from .onboard_adapter import prepare_adapters, qualify
from .onboard_replay import resolve_inside
from .platform import LocalPlatform
from .storage import Evidence, atomic, canonical, digest

BUNDLE = {"predictor.py", "scorer.py", "predict-input.json", "score-input.json", "split.json"}


def connect(root: Path, interpreter: Path, wait_seconds: float = 35) -> dict:
    root = root.resolve(strict=True)
    interpreter = interpreter.resolve(strict=True)
    path = root / "adapter-qualification.json"
    record = json.loads(path.read_bytes()) if path.exists() else None
    if not record or record["phase"] != "adapter_validated" or record["interpreter"] != str(interpreter):
        prepare_adapters(root)
        record = qualify(root, interpreter)
        if record["phase"] != "adapter_validated":
            return record
    persist_path = root / "bridge-dispatch.json"
    with FileLock(str(root / ".bridge-launch.lock"), timeout=0):
        return _connect(root, record, persist_path, wait_seconds)


def _connect(root: Path, ref: dict, path: Path, wait_seconds: float):
    store = Evidence(root)
    qualified = store.get(ref["evidence"])["content"]
    if qualified["phase"] != "adapter_validated" or qualified["interpreter"] != ref["interpreter"]:
        raise ValueError("A verified qualification certificate is required")
    bundle = Path(qualified["bundle"])
    files = {}
    if set(qualified["files"]) != BUNDLE:
        raise ValueError("Unexpected qualified bundle")
    for file, expected in qualified["files"].items():
        raw = resolve_inside(bundle, file).read_bytes()
        if digest(raw) != expected:
            raise ValueError(f"Qualified bridge artifact changed: {file}")
        files[file] = raw
    certificate_id = ref["evidence"]
    dispatch = json.loads(path.read_bytes()) if path.exists() else {"qualification": certificate_id, "receipts": {}}
    if dispatch["qualification"] != certificate_id:
        raise ValueError("Existing dispatch belongs to another qualified bridge")

    launch_root = root / "bridge-dispatch" / certificate_id
    repository = launch_root / "source"
    repository.mkdir(parents=True, exist_ok=True)
    for file, raw in files.items():
        if not (repository / file).is_file() or (repository / file).read_bytes() != raw:
            atomic(repository / file, raw)
    if not (repository / ".git").exists():
        subprocess.run(["git", "init", "-q", "--template=", str(repository)], check=True, timeout=20)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null",
                                        *args], text=True, timeout=20).strip()

    git("add", "--", *sorted(files))
    if git("diff", "--cached", "--name-only"):
        git("-c", "user.name=LOOP Research", "-c", "user.email=local@invalid", "commit", "-qm",
            "Freeze worker-authored integration adapter and admitted data")
    revision = git("rev-parse", "HEAD")
    if set(git("ls-tree", "--name-only", "HEAD").splitlines()) != BUNDLE:
        raise ValueError("Unexpected committed files in bridge repository")
    config = CommandExperiment(repository=str(repository), revision=revision, command=["isolated-bridge"],
                               bridge_interpreter=qualified["interpreter"], split_manifest="split.json",
                               baseline={"score_arm": "physical"}, choices={"score_arm": ["physical", "pattern"]})
    source_file = launch_root / "dispatch-source.json"
    if source_file.exists():
        saved = json.loads(source_file.read_bytes())
        if saved["config"] != config.model_dump():
            raise ValueError("Cannot change a dispatched bridge configuration")
        source = saved["source"]
    else:
        source = prepare(launch_root, config)
        atomic(source_file, canonical({"config": config.model_dump(), "source": source}))
    platform = LocalPlatform(launch_root, config, source)
    deadline = time.monotonic() + wait_seconds
    for arm in ("physical", "pattern"):
        if arm not in dispatch["receipts"]:
            # Repeated submission with the same request reopens the same receipt.
            dispatch["receipts"][arm] = platform.submit(f"bridge-integration:{certificate_id}:{arm}",
                                                       canonical({"score_arm": arm}).decode(), 42)
            atomic(path, canonical(dispatch))
        receipt = dispatch["receipts"][arm]
        result = platform.poll(receipt)
        while result["status"] in {"running", "unknown"} and time.monotonic() < deadline:
            if result["status"] == "unknown":
                result = platform.reconcile(receipt)
            if result["status"] not in {"running", "unknown"}:
                break
            time.sleep(0.5)
            result = platform.poll(receipt)
        if result["status"] in {"running", "unknown"}:
            dispatch["phase"] = "bridge_job_pending"
            atomic(path, canonical(dispatch))
            return dispatch
        expected = qualified["checks"][arm]["metrics"]
        if result["status"] != "succeeded":
            dispatch.update(phase="bridge_job_failed", failed_arm=arm, failure=result)
            atomic(path, canonical(dispatch))
            return dispatch
        actual = result["metrics"]
        if set(actual) != set(expected) or any(
            not math.isclose(value, actual[key], rel_tol=0.0, abs_tol=1e-12) for key, value in expected.items()
        ):
            raise ValueError(f"Dispatched {arm} measurement differs from its qualified reference")
        dispatch.setdefault("measurements", {})[arm] = {
            "evidence": store.put("bridge_integration_measurement", {"arm": arm, "receipt": receipt, "result": result,
                                                                    "run_root": str(launch_root)}, [certificate_id]),
            "metrics": actual,
        }
        atomic(path, canonical(dispatch))
    dispatch.update(phase="bridge_integration_passed", revision=revision,
                    meaning="Known historical controls sent through the normal job backend; no new scientific claim.")
    atomic(path, canonical(dispatch))
    return dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--interpreter", type=Path, required=True)
    args = parser.parse_args()
    record = connect(args.root, args.interpreter)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
