"""Run a prediction/scoring bridge with split inputs and no project or network access."""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .onboard_replay import ReplayRecipe, sandbox
from .storage import atomic, canonical


def read_output(path: Path, limit: int = 24_000_000) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Missing or oversized regular output: {path.name}")
    data = json.loads(path.read_bytes())
    canonical(data)  # Reject NaN and infinities even inside nested evidence.
    if not isinstance(data, dict):
        raise ValueError(f"Output must be an object: {path.name}")
    return data


def isolated_python(interpreter, scratch, script, arguments, logs, *, project=None, mounts=(), timeout=90):
    logs.mkdir(parents=True, exist_ok=True)
    virtual = project or Path("/loop-task")
    recipe = ReplayRecipe(working_directory=".", copy_files=[], read_only_directories=list(mounts),
                          seed_directories=[], steps=[], evidence_files=[], prerequisites=[])
    command = sandbox(interpreter, scratch, virtual, recipe)
    with (logs / "stdout.log").open("wb") as stdout, (logs / "stderr.log").open("wb") as stderr:
        completed = subprocess.run([*command, script, *arguments], stdout=stdout, stderr=stderr,
                                   stdin=subprocess.DEVNULL, timeout=timeout, check=False)
    if completed.returncode:
        raise ValueError(f"{script} failed ({completed.returncode}); see {logs / 'stderr.log'}")


def run_bridge(checkout: Path, interpreter: Path, parameters: dict, split_manifest: dict, logs: Path, timeout: int):
    """Give the predictor features only and the scorer labels plus candidate predictions."""
    expected_ids = [fold["id"] for fold in split_manifest["folds"]]
    features = read_output(checkout / "predict-input.json")
    ground_truth = read_output(checkout / "score-input.json")
    if [f["id"] for f in features["folds"]] != expected_ids or [f["id"] for f in ground_truth["folds"]] != expected_ids:
        raise ValueError("Bridge inputs differ from admitted folds")
    with TemporaryDirectory(prefix="bridge-", dir=logs) as temporary:
        root = Path(temporary)
        prediction = root / "prediction"
        scoring = root / "scoring"
        prediction.mkdir()
        scoring.mkdir()

        def stage(target, script, input_name, data):
            atomic(target / script, (checkout / script).read_bytes())
            atomic(target / input_name, canonical(data))

        stage(prediction, "predictor.py", "features.json", features)
        atomic(prediction / "parameters.json", canonical(parameters))
        isolated_python(interpreter, prediction, "predictor.py",
                        ["features.json", "parameters.json", "predictions.json"], logs / "prediction", timeout=timeout)
        predictions = read_output(prediction / "predictions.json")
        validate_predictions(predictions, features)
        atomic(logs / "raw-predictions.json", canonical(predictions))

        stage(scoring, "scorer.py", "labels.json", ground_truth)
        atomic(scoring / "predictions.json", canonical(predictions))
        isolated_python(interpreter, scoring, "scorer.py", ["labels.json", "predictions.json", "metrics.json"],
                        logs / "scoring", timeout=timeout)
        return read_output(scoring / "metrics.json")


def validate_predictions(predictions, features):
    if set(predictions) != {"folds"} or not isinstance(predictions["folds"], list):
        raise ValueError("Predictor must return a list of folds")
    if len(predictions["folds"]) != len(features["folds"]):
        raise ValueError("Missing prediction folds")
    for predicted, source in zip(predictions["folds"], features["folds"], strict=True):
        if set(predicted) != {"id", "proposals"} or predicted["id"] != source["id"]:
            raise ValueError("Predictor changed the fold IDs")
        videos = {video["video_name"]: video for video in source["videos"]}
        seen = set()
        for proposal in predicted["proposals"]:
            if set(proposal) != {"video_name", "start_frame", "end_frame", "score", "type_idx"}:
                raise ValueError("Predictor changed the proposal schema")
            video = videos.get(proposal["video_name"])
            start, end = proposal["start_frame"], proposal["end_frame"]
            score = proposal["score"]
            if (not video or type(start) is not int or type(end) is not int
                    or not 0 <= start < end < video["nframes"] or type(score) not in {int, float}
                    or not math.isfinite(score) or proposal["type_idx"] != 2):
                raise ValueError("Invalid proposal outside the admitted input")
            identity = (proposal["video_name"], start, end)
            if identity in seen:
                raise ValueError("Duplicate proposal")
            seen.add(identity)
