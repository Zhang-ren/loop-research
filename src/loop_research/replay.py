from __future__ import annotations

from uuid import uuid4

from .engine import Research
from .platform import measure_command
from .training import measure


def reproduce(engine: Research, evidence_id: str) -> dict:
    """Repeat recorded code/data/split, never call a model to regenerate code."""
    engine.references()
    engine.evidence.verify_tree(evidence_id)
    item = engine.evidence.get(evidence_id)
    if item["kind"] not in {"training_result", "holdout_result", "measured_baseline"}:
        raise ValueError("Only externally measured experiments can be reproduced")
    original = item["content"]
    if original.get("status", "succeeded") != "succeeded":
        raise ValueError("Cannot reproduce a non-successful metric result")
    codes = [engine.evidence.get(p)["content"]["code"] for p in item["parents"]
             if engine.evidence.get(p)["kind"] == "model_code"]
    if len(codes) != 1:
        raise ValueError("Measurement must refer to one unambiguous original model")
    if engine.command:
        result = measure_command(engine.root, engine.command, engine.platform.source, codes[0], original["seed"],
                                 original["split"], engine.root / "replays" / uuid4().hex)
    else:
        result = measure(codes[0], original["seed"], split=original["split"])
    differences = {key: result["metrics"].get(key, 0) - value for key, value in original["metrics"].items()
                   if key != "fit_seconds"}
    checks = {"same_code": result["code_sha256"] == original["code_sha256"],
              "same_data": result["data_sha256"] == original["data_sha256"],
              "same_evaluation_samples": result["evaluation_indices"] == original["evaluation_indices"],
              "same_metrics": result["metrics"].keys() == original["metrics"].keys()
              and all(delta == 0 for delta in differences.values())}
    if any(key in original for key in ("predictions", "scores")):
        checks["same_predictions"] = all(result.get(key) == original.get(key) for key in ("predictions", "scores"))
    if "fold_results" in original:
        checks["same_fold_results"] = result.get("fold_results") == original["fold_results"]
    record = {"original": evidence_id, "checks": checks, "passed": all(checks.values()),
              "metric_differences": differences, "result": result,
              "purpose": "reproduction audit, not a new search candidate", "timing_excluded": True}
    artifact = engine.evidence.put("reproduction", record, [evidence_id])
    return {**record, "evidence": artifact}
