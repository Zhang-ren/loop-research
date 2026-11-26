"""Run a fixed local experiment command from a pinned Git snapshot."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path, PurePosixPath

from .models import CommandExperiment
from .storage import atomic, canonical, digest
from .runtime_failure import failed_result


def checked_parameters(code: str, config: CommandExperiment) -> dict:
    parameters = json.loads(code)
    if not isinstance(parameters, dict) or parameters.keys() != config.choices.keys():
        raise ValueError("Candidate must provide exactly the declared parameters")
    for name, value in parameters.items():
        if not any(type(value) is type(option) and value == option for option in config.choices[name]):
            raise ValueError(f"Undeclared value for {name}")
    return parameters


def prepare(root: Path, config: CommandExperiment) -> dict:
    source = Path(config.repository).expanduser().resolve(strict=True)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "--verify", "--end-of-options", config.revision + "^{commit}"],
        text=True, timeout=30,
    ).strip()
    snapshot = root / "source.tar"
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "-C", str(source), "archive", "--format=tar", revision],
                       stdout=archive, check=True, timeout=120)
        archive.seek(0)
        atomic(snapshot, archive.read())
    try:
        splits = read_splits(snapshot, config.split_manifest)
    except Exception:
        snapshot.unlink()
        raise
    return {"revision": revision, "snapshot_sha256": digest(snapshot.read_bytes()), "splits": splits}


def read_splits(snapshot: Path, relative_manifest: str) -> dict:
    path = PurePosixPath(relative_manifest)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Split manifest must be a path inside the committed repository")
    with tarfile.open(snapshot) as archive:
        member = archive.getmember(str(path))
        if not member.isfile():
            raise ValueError("Split manifest must be a regular committed file")
        with archive.extractfile(member) as stream:
            splits = json.load(stream)
    if isinstance(splits, dict) and set(splits) == {"folds", "heldout"}:
        folds = splits["folds"]
        if not isinstance(folds, list) or not folds:
            raise ValueError("A grouped protocol requires at least one fold")
        fold_ids = set()
        historical_ids = set()
        for fold in folds:
            if not isinstance(fold, dict) or set(fold) != {"id", "train", "validation"}:
                raise ValueError("Every fold must declare its ID, train and validation IDs")
            fold_id = fold["id"]
            if not isinstance(fold_id, str) or not fold_id or fold_id in fold_ids:
                raise ValueError("Fold IDs must be unique nonempty strings")
            fold_ids.add(fold_id)
            historical_ids.update(disjoint_ids((fold["train"], fold["validation"])))
        heldout = splits["heldout"]
        if heldout is not None:
            if not isinstance(heldout, dict) or set(heldout) != {"train", "test"}:
                raise ValueError("Independent holdout requires its own train and test IDs")
            disjoint_ids(heldout.values())
            if historical_ids & {str(x) for x in heldout["test"]}:
                raise ValueError("Independent test IDs overlap the explored folds")
        return splits
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation and test")
    disjoint_ids(splits.values())
    return splits


def disjoint_ids(groups):
    seen = set()
    for group in groups:
        if not isinstance(group, list) or not group or any(type(x) not in (str, int) for x in group):
            raise ValueError("Every split must contain nonempty string or integer sample IDs")
        unique = {str(x) for x in group}
        if len(unique) != len(group) or seen.intersection(unique):
            raise ValueError("Split manifests must contain unique, disjoint sample IDs")
        seen.update(unique)
    return seen


def split_view(splits, split):
    if "folds" in splits:
        if split == "validation":
            def coordinates(group):
                return [canonical([fold["id"], sample]).decode() for fold in splits["folds"] for sample in fold[group]]
            return {"folds": splits["folds"]}, coordinates("train"), coordinates("validation")
        if not splits["heldout"]:
            raise ValueError("This grouped study has no independent holdout")
        heldout = splits["heldout"]
        return heldout, heldout["train"], heldout["test"]
    return {"train": splits["train"], split: splits[split]}, splits["train"], splits[split]


def run_command(root: Path, config: CommandExperiment, source: dict, code: str, seed: int,
                split: str, logs: Path) -> dict:
    if split not in {"validation", "test"}:
        raise ValueError("Only validation and test evaluation are supported")
    parameters = checked_parameters(code, config)
    snapshot = root / "source.tar"
    if source["snapshot_sha256"] != digest(snapshot.read_bytes()):
        raise ValueError("Committed experiment snapshot changed")
    logs.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checkout-", dir=logs) as temporary:
        directory = Path(temporary)
        checkout = directory / "repo"
        checkout.mkdir()
        with tarfile.open(snapshot) as archive:
            if any(not (m.isfile() or m.isdir()) for m in archive.getmembers()):
                raise ValueError("Experiment snapshot may contain only regular files and directories")
            archive.extractall(checkout, filter="data")
        manifest = directory / "split.json"
        # Search iterations receive only train and validation IDs.
        visible, train_indices, evaluation_indices = split_view(source["splits"], split)
        atomic(manifest, canonical(visible))
        (checkout / config.split_manifest).unlink()
        params_path, metrics_path = directory / "parameters.json", directory / "metrics.json"
        atomic(params_path, canonical(parameters))
        tmp = directory / "tmp"
        tmp.mkdir()
        env = {"PATH": os.defpath + ":" + str(Path(sys.executable).parent), "HOME": str(directory),
               "TMPDIR": str(tmp), "LANG": "en_US.UTF-8",
               "LOOP_PARAMETERS": str(params_path), "LOOP_METRICS": str(metrics_path),
               "LOOP_SPLIT_MANIFEST": str(manifest), "LOOP_SPLIT": split, "LOOP_SEED": str(seed)}
        if config.bridge_interpreter:
            from .bridge_runtime import run_bridge
            if set(visible) != {"folds"}:
                raise ValueError("This historical bridge is limited to the admitted exploration folds")
            result = run_bridge(checkout, Path(config.bridge_interpreter).resolve(strict=True), parameters,
                                visible, logs, config.timeout_seconds)
            atomic(metrics_path, canonical(result))
        else:
            with (logs / "command.stdout.log").open("wb") as out, (logs / "command.stderr.log").open("wb") as err:
                subprocess.run(config.command, cwd=checkout, env=env, stdin=subprocess.DEVNULL,
                               stdout=out, stderr=err, check=True, timeout=config.timeout_seconds)
        if not metrics_path.is_file() or metrics_path.is_symlink() or metrics_path.stat().st_size > 4_000_000:
            raise ValueError("Command must write a small regular JSON metrics file")
        output = json.loads(metrics_path.read_bytes())
        if not isinstance(output, dict) or set(output) - {"metrics", "predictions", "scores", "fold_results"}:
            raise ValueError("Invalid measurement output fields")
        metrics = output.get("metrics")
        if not isinstance(metrics, dict) or not metrics or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values()
        ):
            raise ValueError("Command returned missing or nonfinite metrics")
        for field in ("predictions", "scores"):
            if field in output and (not isinstance(output[field], list) or len(output[field]) != len(evaluation_indices)):
                raise ValueError(f"{field} must align with the evaluated sample IDs")
        if "folds" in visible:
            folds = output.get("fold_results", [])
            if not isinstance(folds, list) or not all(isinstance(fold, dict) for fold in folds):
                raise ValueError("Grouped experiment result must contain fold_results")
            if [fold.get("id") for fold in folds] != [fold["id"] for fold in visible["folds"]]:
                raise ValueError("Fold results must match the protocol order and IDs")
        atomic(logs / "metrics.json", canonical(output))
    return {**output, "split": split, "seed": seed, "code_sha256": digest(code.encode()),
            "data_sha256": digest(canonical([source["snapshot_sha256"], source["splits"]])),
            "evaluation_indices": evaluation_indices, "train_indices": train_indices,
            "source_revision": source["revision"], "logs": str(logs)}


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory / "request.json").read_bytes())
    request_id = digest(canonical(request))
    try:
        result = run_command(directory.parent.parent, CommandExperiment.model_validate(request["command_experiment"]),
                             request["source"], request["code"], request["seed"], request["split"], directory)
        result["status"] = "succeeded"
    except Exception as exc:
        result = failed_result(exc, traceback.format_exc())
    atomic(directory / "result.json", canonical({**result, "request_sha256": request_id}))


if __name__ == "__main__":
    main()
