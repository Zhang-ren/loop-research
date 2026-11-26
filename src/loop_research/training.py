"""Frozen local evaluation adapter. Agents only return a bounded model factory.

This example uses a small, reversible classification benchmark, not online business traffic.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from .storage import atomic, canonical, digest
from .runtime_failure import failed_result

BASELINE_CODE = '''from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

def build_model(seed):
    return make_pipeline(LogisticRegression(C=1.0, max_iter=500, random_state=seed))
'''

ALLOWED_IMPORTS = {
    "sklearn.pipeline": {"make_pipeline", "Pipeline"},
    "sklearn.preprocessing": {"StandardScaler", "RobustScaler", "MinMaxScaler", "PolynomialFeatures"},
    "sklearn.linear_model": {"LogisticRegression", "RidgeClassifier"},
    "sklearn.ensemble": {"RandomForestClassifier", "ExtraTreesClassifier", "HistGradientBoostingClassifier"},
    "sklearn.svm": {"SVC", "LinearSVC"},
    "sklearn.feature_selection": {"SelectKBest", "f_classif"},
    "sklearn.decomposition": {"PCA"},
}


def validate_code(code: str):
    if len(code.encode()) > 20000:
        raise ValueError("Model factory exceeds size budget")
    tree = ast.parse(code)
    imported = set()
    functions = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise ValueError("Import outside example protocol")
            for alias in node.names:
                if alias.asname or alias.name not in ALLOWED_IMPORTS[node.module]:
                    raise ValueError("Unsupported import")
                imported.add(alias.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            raise ValueError("Only imports and build_model are editable")
    if len(functions) != 1:
        raise ValueError("Exactly one model factory required")
    fn = functions[0]
    if fn.name != "build_model" or fn.decorator_list or fn.returns or fn.type_params:
        raise ValueError("Invalid factory")
    if (len(fn.args.args) != 1 or fn.args.args[0].arg != "seed" or fn.args.args[0].annotation
            or fn.args.posonlyargs or fn.args.kwonlyargs or fn.args.vararg or fn.args.kwarg or fn.args.defaults):
        raise ValueError("Factory accepts seed only")
    if len(fn.body) != 1 or not isinstance(fn.body[0], ast.Return):
        raise ValueError("Factory must be a single estimator expression")
    permitted = (ast.Call, ast.Name, ast.Load, ast.Constant, ast.keyword, ast.Tuple, ast.List,
                 ast.UnaryOp, ast.USub, ast.Dict)
    for node in ast.walk(fn.body[0].value):
        if not isinstance(node, permitted):
            raise ValueError(f"Unsupported model expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in imported | {"seed"}:
            raise ValueError("Unknown name")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in imported):
            raise ValueError("Only whitelisted constructors callable")
        if isinstance(node, ast.keyword) and node.arg is None:
            raise ValueError("Keyword expansion forbidden")
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)) and abs(node.value) > 10000:
            raise ValueError("Example compute bound exceeded")
    return tree


def measure(code: str, seed: int, *, split="validation") -> dict:
    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    tree = validate_code(code)
    namespace = {}
    exec(compile(tree, "<approved-model-factory>", "exec"), namespace)
    data = load_breast_cancer()
    data_hash = hashlib.sha256(data.data.tobytes() + data.target.tobytes()).hexdigest()
    indices = np.arange(len(data.target))
    train, held = train_test_split(indices, test_size=0.4, stratify=data.target, random_state=seed)
    validation, test = train_test_split(held, test_size=0.5, stratify=data.target[held], random_state=seed)
    evaluate = validation if split == "validation" else test
    model = namespace["build_model"](seed)
    started = time.monotonic()
    model.fit(data.data[train], data.target[train])
    predicted = model.predict(data.data[evaluate])
    probabilities = model.predict_proba(data.data[evaluate])[:, 1] if hasattr(model, "predict_proba") else None
    scores = probabilities if probabilities is not None else model.decision_function(data.data[evaluate])
    metrics = {"accuracy": float(accuracy_score(data.target[evaluate], predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(data.target[evaluate], predicted)),
               "roc_auc": float(roc_auc_score(data.target[evaluate], scores)),
               "fit_seconds": time.monotonic() - started}
    if probabilities is not None:
        metrics["brier_loss"] = float(brier_score_loss(data.target[evaluate], probabilities))
    return {"metrics": metrics, "data_sha256": data_hash, "code_sha256": digest(code.encode()),
            "split": split, "seed": seed, "train_indices": train.tolist(), "evaluation_indices": evaluate.tolist(),
            "labels": data.target[evaluate].tolist(), "predictions": predicted.tolist(), "scores": scores.tolist()}


def main():
    directory = Path(sys.argv[1]).resolve()
    request = json.loads((directory / "request.json").read_bytes())
    try:
        result = measure(request["code"], request["seed"], split=request.get("split", "validation"))
        result.update(status="succeeded", request_sha256=digest(canonical(request)))
    except Exception as exc:
        result = {**failed_result(exc, traceback.format_exc()), "request_sha256": digest(canonical(request))}
    atomic(directory / "result.json", canonical(result))


if __name__ == "__main__":
    main()
