"""Run a fit, held prediction and evaluator in separate filesystem views."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import Field

from .bridge_runtime import read_output
from .models import StrictModel
from .onboard_replay import resolve_inside
from .sealed_stage import file_hash, stage_command
from .storage import atomic, canonical


class Video(StrictModel):
    file: str
    subject: str = Field(min_length=1)
    frames: int = Field(ge=1)

    @property
    def stem(self):
        name = Path(self.file)
        if not name.stem or name.name != self.file or name.suffix != ".npz" or name.stem.startswith("."):
            raise ValueError(f"Invalid video filename: {self.file}")
        return name.stem


class Fold(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    train_files: list[str] = Field(min_length=1)
    held_files: list[str] = Field(min_length=1)
    duration_frames: float = Field(gt=0, allow_inf_nan=False)


class Projection(StrictModel):
    files: dict[str, str]
    feature_seal_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    videos: list[Video] = Field(min_length=1)
    folds: list[Fold] = Field(min_length=1)

    @classmethod
    def verified(cls, root: Path, expected: str, split: dict):
        manifest = resolve_inside(root, "manifest.json")
        if file_hash(manifest) != expected:
            raise ValueError("Fold projection changed after admission")
        data = cls.model_validate_json(manifest.read_bytes())
        videos = {row.file: row for row in data.videos}
        assets = {asset(kind, row) for row in data.videos for kind in ("features", "supervision", "truth")}
        if len(videos) != len(data.videos) or set(data.files) not in (assets, assets | {"split.json"}):
            raise ValueError("Projection must account for exactly the admitted assets")
        if [f.id for f in data.folds] != [f["id"] for f in split["folds"]]:
            raise ValueError("Projection fold IDs differ from the frozen split")
        held = []
        for fold, admitted in zip(data.folds, split["folds"], strict=True):
            train, validation = set(fold.train_files), set(fold.held_files)
            if (len(train) != len(fold.train_files) or len(validation) != len(fold.held_files)
                    or train & validation or train | validation != set(videos)
                    or {videos[name].subject for name in train} != set(admitted["train"])
                    or {videos[name].subject for name in validation} != set(admitted["validation"])):
                raise ValueError(f"Fold {fold.id} differs from the frozen subject separation")
            held.extend(fold.held_files)
        if len(held) != len(videos) or set(held) != set(videos):
            raise ValueError("Every video must be held out exactly once")
        for path, checksum in data.files.items():
            if file_hash(resolve_inside(root, path)) != checksum:
                raise ValueError(f"Projected input changed: {path}")
        return data


def asset(kind: str, row: Video) -> str:
    suffix = ".json" if kind == "truth" else ".npz"
    return f"{kind}/{row.stem}{suffix}"


def run_folds(checkout: Path, interpreter: Path, data_dir: Path, manifest_sha256: str,
              parameters: dict, seed: int, split: dict, logs: Path, timeout: int):
    """Project adapters are frozen in checkout. Only per-role inputs are mounted."""
    projection = Projection.verified(data_dir, manifest_sha256, split)
    by_file = {video.file: video for video in projection.videos}

    def run(script, directory, inputs, config, args, result):
        with TemporaryDirectory(prefix=script.removesuffix(".py") + "-", dir=logs) as temporary:
            scratch = Path(temporary)
            (scratch / script).write_bytes(resolve_inside(checkout, script).read_bytes())
            for name, value in config.items():
                atomic(scratch / name, canonical(value))
            command = stage_command(interpreter, scratch, Path("/loop-task"), script, args,
                                    readonly_files=inputs)
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
                completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                           timeout=timeout, check=False)
            if completed.returncode:
                raise ValueError(f"{script} failed ({completed.returncode}); see {directory / 'stderr.log'}")
            return result(scratch)

    def preserve(scratch, source, destination):
        path = resolve_inside(scratch, source)
        if not path.is_file() or not 0 < path.stat().st_size <= 512_000_000:
            raise ValueError(f"Missing or oversized regular artifact: {source}")
        path.replace(destination)
        return destination

    score_inputs, score_folds = {}, []
    for fold in projection.folds:
        fold_dir = logs / fold.id
        common = {"id": fold.id, "seed": seed, "duration_frames": fold.duration_frames}
        train = [by_file[filename] for filename in fold.train_files]
        held = [by_file[filename] for filename in fold.held_files]

        def input_set(kinds, videos):
            paths = [asset(kind, video) for video in videos for kind in kinds]
            return {path: resolve_inside(data_dir, path) for path in paths}

        def collect_fit(scratch):
            atomic(fold_dir / "fit.json", canonical(read_output(scratch / "fit.json")))
            return preserve(scratch, "model.npz", fold_dir / "model.npz")

        model = run("fit.py", fold_dir / "fit-logs", input_set(("features", "supervision"), train),
                    {"fold.json": {**common, "videos": [v.model_dump() for v in train]},
                     "parameters.json": parameters}, ["fold.json", "parameters.json", "model.npz", "fit.json"],
                    collect_fit)
        predictions = run("predict.py", fold_dir / "predict-logs", {**input_set(("features",), held),
                          "model.npz": model}, {"fold.json": {**common, "videos": [v.model_dump() for v in held]}},
                          ["fold.json", "model.npz", "predictions.npz"],
                          lambda job: preserve(job, "predictions.npz", fold_dir / "predictions.npz"))
        path = f"predictions/{fold.id}.npz"
        score_inputs.update(input_set(("truth",), held))
        score_inputs[path] = predictions
        score_folds.append({**common, "videos": [v.model_dump() for v in held], "predictions": path})
    return run("score_folds.py", logs / "scoring", score_inputs, {"folds.json": {"folds": score_folds}},
               ["folds.json", "metrics.json"], lambda job: read_output(job / "metrics.json"))
