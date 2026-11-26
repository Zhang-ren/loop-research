"""Score a threshold classifier on a fixed synthetic sensor benchmark."""
import json
import os
from pathlib import Path

parameters = json.loads(Path(os.environ["LOOP_PARAMETERS"]).read_text())
samples = json.loads(Path("samples.json").read_text())
splits = json.loads(Path(os.environ["LOOP_SPLIT_MANIFEST"]).read_text())
split = os.environ["LOOP_SPLIT"]
evaluated = splits[split]
predictions = [int(samples[key][0] >= parameters["threshold"]) for key in evaluated]
labels = [samples[key][1] for key in evaluated]
accuracy = sum(a == b for a, b in zip(labels, predictions, strict=True)) / len(evaluated)
Path(os.environ["LOOP_METRICS"]).write_text(json.dumps({
    "metrics": {"accuracy": accuracy}, "predictions": predictions,
}))
