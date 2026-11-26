"""Evidence-linked follow-up plans for command experiments."""
from __future__ import annotations

import json

from .command_candidate import candidate_code, candidate_parts
from .refutations import active_rule
from .storage import digest


def history(engine):
    for path in sorted((engine.root / "batches").glob("*.json"), reverse=True):
        key = json.loads(path.read_bytes())["evidence"]
        yield key, engine.evidence.get(key)["content"]


def exact_refuted(engine, code):
    protocol = engine.references()["protocol"]
    code_hash = digest(code.encode())
    for path in (engine.root / "refutations").glob("*.json"):
        decision = engine.evidence.get(json.loads(path.read_bytes())["evidence"])["content"]
        if decision["protocol"] == protocol and decision["candidate_sha256"] == code_hash:
            return True
    return False


def refuted(engine, code):
    return exact_refuted(engine, code) or active_rule(engine, code) is not None


def propose(engine, state):
    """Select a novel combination or the outermost unmeasured numerical value."""
    batches = list(history(engine))
    if not batches or batches[0][1]["tasks"][0].get("mode", "candidate") != "candidate":
        return None
    current_code = engine.evidence.get(state.best_code)["content"]["code"]
    current, model = candidate_parts(current_code, engine.command)
    measured_codes = {current_code}
    for parent in engine.evidence.get(state.baseline_evidence)["parents"]:
        item = engine.evidence.get(parent)
        if item["kind"] == "model_code":
            measured_codes.add(item["content"]["code"])
    for _, batch in batches:
        for task in batch["tasks"]:
            if task.get("code_artifact") and task["status"] in {"succeeded", "code_failed"}:
                measured_codes.add(engine.evidence.get(task["code_artifact"])["content"]["code"])

    batch_id, previous = batches[0]
    if not previous.get("baseline"):
        return None
    baseline_metrics = engine.evidence.get(previous["baseline"])["content"]["metrics"]
    improvements = {}
    for task in previous["tasks"]:
        if task["status"] != "succeeded" or not engine.dominates(task["metrics"], baseline_metrics):
            continue
        target = task["hypothesis"]["machine_target"]
        if not target.startswith("parameter."):
            continue
        parameters, measured_model = candidate_parts(
            engine.evidence.get(task["code_artifact"])["content"]["code"], engine.command)
        if measured_model == model:
            name = target.removeprefix("parameter.")
            improvements[name] = (parameters[name], task["evidence"])
    if len(improvements) >= 2:
        selected = sorted(improvements)[:2]
        combined = {**current, **{name: improvements[name][0] for name in selected}}
        code = candidate_code(combined, model)
        if code not in measured_codes and not refuted(engine, code):
            return ("combination", code, [batch_id, *(improvements[name][1] for name in selected)],
                    {"combined_parameters": selected})

    for name, allowed in sorted(engine.command.choices.items()):
        if type(current[name]) not in (float, int):
            continue
        # Use prior negative boundary measurements only with identical other parameters.
        local_lower, local_upper = min(allowed), max(allowed)
        parents = [batch_id, state.best_evidence]
        for _, batch in batches:
            for task in batch["tasks"]:
                if task.get("mode") != "probe" or task["status"] != "succeeded":
                    continue
                probe, probe_model = candidate_parts(
                    engine.evidence.get(task["code_artifact"])["content"]["code"], engine.command)
                if probe_model != model or any(probe[k] != current[k] for k in current if k != name):
                    continue
                if not engine.dominates(state.best_metrics, task["metrics"]):
                    continue
                if probe[name] < current[name]:
                    local_lower = max(local_lower, min(x for x in allowed if x > probe[name]))
                elif probe[name] > current[name]:
                    local_upper = min(local_upper, max(x for x in allowed if x < probe[name]))
                else:
                    continue
                parents.append(task["evidence"])
        candidates = sorted((x for x in allowed if local_lower <= x <= local_upper),
                            key=lambda x: (-abs(x - current[name]), x))
        for value in candidates:
            code = candidate_code({**current, name: value}, model)
            if code not in measured_codes and not refuted(engine, code):
                return ("probe", code, list(dict.fromkeys(parents)),
                        {"parameter": name, "declared_range": [min(allowed), max(allowed)],
                         "local_range": [local_lower, local_upper], "tested_value": value})
    return None
