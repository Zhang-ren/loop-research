"""Evaluate a mutable model with a sealed predictor, scorer and per-split inputs."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory

from .bridge_runtime import isolated_python, read_output
from .command_candidate import candidate_parts, snapshot_file
from .command_experiment import split_view
from .models import CommandExperiment
from .sealed_stage import file_hash
from .storage import atomic, canonical, digest
from .runtime_failure import failed_result


def scalar(value):
    return type(value) in (str, int) or (type(value) is float and math.isfinite(value))


def measure_prediction(root, config, source, code, seed, split, logs):
    if split not in {"validation", "test"} or "folds" in source["splits"]:
        raise ValueError("Isolated prediction needs a train/validation/test manifest")
    parameters, model = candidate_parts(code, config)
    snapshot = root / "source.tar"
    if file_hash(snapshot) != source["snapshot_sha256"]:
        raise ValueError("Committed experiment snapshot changed")
    _, train_ids, eval_ids = split_view(source["splits"], split)
    interpreter = Path(config.bridge_interpreter).resolve(strict=True)
    visible_roots = [interpreter.parent.parent, *(Path(p).resolve() for p in ("/usr", "/bin", "/lib", "/lib64"))]
    private_roots = (root.resolve(), Path(config.repository).resolve())
    if any(private.is_relative_to(visible) for private in private_roots for visible in visible_roots):
        raise ValueError("Isolated prediction requires the run and repository outside mounted runtime directories")
    with tarfile.open(snapshot) as archive:
        samples = json.loads(snapshot_file(archive, "samples.json", 32_000_000))
        predictor, scorer = (snapshot_file(archive, name, 128_000) for name in ("predict.py", "score.py"))
    expected = {str(key) for group in source["splits"].values() for key in group}
    if not isinstance(samples, dict) or samples.keys() != expected or any(
        not isinstance(row, dict) or set(row) != {"features", "label"} or not isinstance(row["features"], list)
        or not scalar(row["label"]) for row in samples.values()
    ):
        raise ValueError("samples.json must contain exactly the frozen sample IDs with features and labels")

    logs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="predict-", dir=logs) as directory:
        work = Path(directory)
        atomic(work / "predict.py", predictor)
        atomic(work / "model.py", model.encode())
        train = [samples[str(key)] for key in train_ids]
        features = [samples[str(key)]["features"] for key in eval_ids]
        atomic(work / "inputs.json", canonical({"train": train, "features": features,
                                                "parameters": parameters, "seed": seed}))
        isolated_python(interpreter, work, "predict.py", ["inputs.json", "predictions.json"],
                        logs / "prediction", timeout=config.timeout_seconds)
        output = read_output(work / "predictions.json")
        predictions = output.get("predictions")
        if (set(output) != {"predictions"} or not isinstance(predictions, list)
                or len(predictions) != len(eval_ids) or not all(map(scalar, predictions))):
            raise ValueError("Predictor must return one scalar prediction per evaluation sample in order")
    with TemporaryDirectory(prefix="score-", dir=logs) as directory:
        work = Path(directory)
        atomic(work / "score.py", scorer)
        atomic(work / "inputs.json", canonical({"labels": [samples[str(key)]["label"] for key in eval_ids],
                                               "predictions": predictions}))
        isolated_python(interpreter, work, "score.py", ["inputs.json", "metrics.json"],
                        logs / "scoring", timeout=config.timeout_seconds)
        scored = read_output(work / "metrics.json")
    metrics = scored.get("metrics")
    if set(scored) != {"metrics"} or not isinstance(metrics, dict) or not metrics or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in metrics.values()
    ):
        raise ValueError("Frozen scorer must return finite metrics")
    result = {"metrics": metrics, "predictions": predictions}
    atomic(logs / "metrics.json", canonical(result))
    return {**result, "split": split, "seed": seed, "code_sha256": digest(code.encode()),
            "data_sha256": digest(canonical([source["snapshot_sha256"], source["splits"]])),
            "evaluation_indices": eval_ids, "train_indices": train_ids,
            "source_revision": source["revision"], "logs": str(logs)}


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory / "request.json").read_bytes())
    try:
        config = CommandExperiment.model_validate(request["command_experiment"])
        result = measure_prediction(directory.parent.parent, config, request["source"], request["code"],
                                    request["seed"], request["split"], directory)
        result["status"] = "succeeded"
    except Exception as exc:
        result = failed_result(exc, traceback.format_exc())
    atomic(directory / "result.json", canonical({**result, "request_sha256": digest(canonical(request))}))


if __name__ == "__main__":
    main()
