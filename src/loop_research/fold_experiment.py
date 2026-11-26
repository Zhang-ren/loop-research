"""Local platform worker for the separated fold training mode."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

from .command_experiment import checked_parameters, run_command, split_view
from .fold_runtime import run_folds
from .models import CommandExperiment
from .sealed_stage import file_hash
from .storage import atomic, canonical, digest
from .runtime_failure import failed_result


def measure_command(root, config, source, code, seed, split, logs):
    """Dispatch a frozen experiment using its original or isolated-fold protocol."""
    if config.command[0] != "isolated-fold":
        return run_command(root, config, source, code, seed, split, logs)
    return measure_folds(root, config, source, code, seed, split, logs)


def measure_folds(root, config, source, code, seed, split, logs):
    if split != "validation" or "folds" not in source["splits"] or source["splits"]["heldout"] is not None:
        raise ValueError("This fold adapter uses the admitted exploration split only")
    parameters = checked_parameters(code, config)
    snapshot = root / "source.tar"
    if file_hash(snapshot) != source["snapshot_sha256"]:
        raise ValueError("Committed experiment snapshot changed")
    visible, train_ids, evaluation_ids = split_view(source["splits"], split)
    data = Path(config.command[1]).resolve(strict=True)
    interpreter = Path(config.bridge_interpreter).resolve(strict=True)
    logs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="adapter-", dir=logs) as temporary:
        checkout = Path(temporary)
        with tarfile.open(snapshot) as archive:
            for name in ("fit.py", "predict.py", "score_folds.py"):
                member = archive.getmember(name)
                if not member.isfile() or member.size > 256_000:
                    raise ValueError(f"Invalid sealed fold adapter: {name}")
                with archive.extractfile(member) as handle:
                    atomic(checkout / name, handle.read())
        output = run_folds(checkout, interpreter, data, config.command[2], parameters, seed,
                           visible, logs, config.timeout_seconds)
    if set(output) != {"metrics", "fold_results"}:
        raise ValueError("Fold scoring requires metrics and fold_results")
    metrics, folds = output["metrics"], output["fold_results"]
    if not isinstance(metrics, dict) or not metrics or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("Fold scorer returned missing or nonfinite metrics")
    if (not isinstance(folds, list) or not all(isinstance(fold, dict) for fold in folds)
            or [fold.get("id") for fold in folds] != [fold["id"] for fold in visible["folds"]]):
        raise ValueError("Fold score results must match the protocol order and IDs")
    atomic(logs / "metrics.json", canonical(output))
    return {**output, "split": split, "seed": seed, "code_sha256": digest(code.encode()),
            "data_sha256": digest(canonical([source["snapshot_sha256"], source["splits"], config.command[2]])),
            "evaluation_indices": evaluation_ids, "train_indices": train_ids,
            "source_revision": source["revision"], "logs": str(logs)}


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory / "request.json").read_bytes())
    key = digest(canonical(request))
    try:
        result = measure_folds(directory.parent.parent, CommandExperiment.model_validate(request["command_experiment"]),
                               request["source"], request["code"], request["seed"], request["split"], directory)
        result["status"] = "succeeded"
    except Exception as exc:
        result = failed_result(exc, traceback.format_exc())
    atomic(directory / "result.json", canonical({**result, "request_sha256": key}))


if __name__ == "__main__":
    main()
