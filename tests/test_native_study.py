import json
import subprocess
import sys
from pathlib import Path

import pytest

from loop_research.command_experiment import prepare
from loop_research.command_candidate import candidate_code, candidate_parts, frozen_baseline_code
from loop_research.models import CommandExperiment, Project
from loop_research.native_study_runtime import NativeAxis, apply_axes, measure_native, validate_mutable_sources
from loop_research.onboard_native import NativeRecipe, run_recipe, validate_recipe
from loop_research.storage import canonical, digest
from test_native_onboarding import require_bubblewrap


def test_native_candidate_uses_documented_workdir_and_one_sealed_view(tmp_path):
    require_bubblewrap()
    bundle = tmp_path / "bundle"
    research = tmp_path / "research"
    bundle.mkdir()
    research.mkdir()
    originals = {
        "src/train.py": (
            "from pathlib import Path\nimport json\n"
            "config = json.loads(Path('params.json').read_text())\n"
            "Path('model').write_text(str(int(Path('train.txt').read_text()) + config['train']['size']))\n"
        ),
        "src/evaluate.py": (
            "from pathlib import Path\nimport json\n"
            "assert not Path('views').exists()\n"
            "value = int(Path('model').read_text()) * int(Path('evaluation.txt').read_text())\n"
            "Path('eval').mkdir()\nPath('eval/metrics.json').write_text(json.dumps({'value': value}))\n"
        ),
        "params.json": '{"train":{"size":2},"split":{"seed":9}}',
    }
    views = {"validation": {"train.txt": b"5", "evaluation.txt": b"3"},
             "test": {"train.txt": b"5", "evaluation.txt": b"9"}}
    for name, text in originals.items():
        target = bundle / "native" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for scope, inputs in views.items():
        for name, content in inputs.items():
            target = bundle / "views" / scope / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    manifest = {"train": ["t"], "validation": ["v"], "test": ["h"]}
    (bundle / "split.json").write_bytes(canonical(manifest))
    recipe = NativeRecipe(
        stages=[{"name": "train", "script": "src/train.py", "outputs": ["model"]},
                {"name": "score", "script": "src/evaluate.py", "outputs": ["eval"]}],
        source_files=list(originals), input_files=["train.txt", "evaluation.txt"],
        working_directories=[{"stage": "train", "path": "."}, {"stage": "score", "path": "."}],
        seed_paths=[], attestations=[],
        readonly_directories=[], metrics_file="eval/metrics.json",
        metrics=[{"name": "value", "keys": ["value"]}], questions=[],
    )
    validate_recipe(recipe, bundle / "native", None,
                    {"baseline_entrypoints": ["src/train.py"], "independent_evaluation_entrypoints": ["src/evaluate.py"]},
                    dict.fromkeys(originals))
    axis = NativeAxis(name="size", config_file="params.json", key_path=["train", "size"],
                      baseline=2, choices=[2, 4], stage=None, argument=None)
    (bundle / "native-study.json").write_bytes(canonical({
        "recipe": recipe.model_dump(), "axes": [axis.model_dump()],
        "view_sha256": {scope: {name: digest(data) for name, data in files.items()} for scope, files in views.items()},
    }))
    subprocess.run(["git", "init", "-q", "--template=", str(bundle)], check=True)
    subprocess.run(["git", "-C", str(bundle), "add", "."], check=True)
    subprocess.run(["git", "-C", str(bundle), "-c", "user.name=test", "-c", "user.email=test@local",
                    "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"], check=True)
    config = CommandExperiment(repository=str(bundle), command=["isolated-native"], split_manifest="split.json",
                               baseline={"size": 2}, choices={"size": [2, 4]}, bridge_interpreter=sys.executable)
    source = prepare(research, config)
    for split, expected, ids in (("validation", 27, ["v"]), ("test", 81, ["h"])):
        logs = research / split
        result = measure_native(research, config, source, '{"size":4}', 9, split, logs)
        assert result["metrics"] == {"value": expected}
        assert result["evaluation_indices"] == ids
        assert [stage["stage"] for stage in result["native_stages"]] == ["train", "score"]
        seed = next((logs / "native-stages").glob("*/input.json"))
        workspace = Path(json.loads(seed.read_bytes())["workspace"])
        assert (workspace / "evaluation.txt").read_bytes() == views[split]["evaluation.txt"]
        assert (workspace / "src/evaluate.py").read_text() == originals["src/evaluate.py"]
        assert json.loads((workspace / "params.json").read_bytes()) == {"train": {"size": 4}, "split": {"seed": 9}}
    assert (bundle / "native/params.json").read_text() == originals["params.json"]


def test_reject_native_axis_on_frozen_split():
    from loop_research.native_study import validate_axes
    axis = NativeAxis(name="seed", config_file="params.yaml", key_path=["prepare", "seed"],
                      baseline=2, choices=[2, 4], stage=None, argument=None)
    recipe = NativeRecipe(stages=[], source_files=["params.yaml"], seed_paths=[], attestations=[],
                          readonly_directories=[], metrics_file=None, metrics=[], questions=[])
    with pytest.raises(ValueError, match="frozen protocol"):
        validate_axes(recipe, [axis], set())


def test_python_axes_bind_original_defaults_and_shared_model_values(tmp_path):
    from loop_research.native_study import validate_axes

    sources = {
        "train.py": "native['Ridge']()\ntrain(metric_weight=1.0)\n",
        "ridge.py": "class Ridge:\n    def __init__(self, ridge_mult=0.001): pass\n",
        "model.py": "PARAMETERS = {'层数': 1, 'dropout': 0.2}  # exact bytes elsewhere\n",
        "score.py": "from model import PARAMETERS\n",
        "prepare.py": "",
    }
    for name, body in sources.items():
        (tmp_path / name).write_bytes(body.encode())
    recipe = NativeRecipe(
        stages=[{"name": name, "script": f"{name}.py", "outputs": [name + ".out"]}
                for name in ("prepare", "train", "score")],
        source_files=list(sources), input_files=[], seed_paths=[], attestations=[], readonly_directories=[],
        metrics_file=None, metrics=[], questions=[],
        stage_views=[
            {"stage": name, "source_files": files, "input_files": [], "previous_outputs": []}
            for name, files in (("prepare", ["prepare.py"]), ("train", ["train.py", "ridge.py", "model.py"]),
                                ("score", ["score.py", "model.py"]))
        ],
    )
    axes = [
        NativeAxis(name="ridge_mult", stage="train", key_path=["ridge_mult"], config_file=None, argument=None,
                   baseline=0.001, choices=[0.001, 0.01],
                   source_binding={"path": "train.py", "kind": "keyword", "target": "native['Ridge']",
                                   "default_source": "ridge.py", "default_callable": "Ridge.__init__"}),
        NativeAxis(name="dropout", stage="train", key_path=["dropout"], config_file=None, argument=None,
                   baseline=0.2, choices=[0.2, 0.4],
                   source_binding={"path": "model.py", "kind": "assignment", "target": "PARAMETERS"}),
    ]
    validate_axes(recipe, axes, {"prepare"})
    apply_axes(tmp_path, recipe, axes, {"ridge_mult": 0.001, "dropout": 0.2}, ["prepare"])
    assert {name: (tmp_path / name).read_bytes() for name in sources} == {
        name: body.encode() for name, body in sources.items()
    }
    with pytest.raises(ValueError, match="outside its sealed choices"):
        apply_axes(tmp_path, recipe, axes, {"ridge_mult": 0.02, "dropout": 0.2}, ["prepare"])
    apply_axes(tmp_path, recipe, axes, {"ridge_mult": 0.01, "dropout": 0.4}, ["prepare"])
    assert (tmp_path / "train.py").read_bytes() == sources["train.py"].replace(
        "Ridge']()", "Ridge'](ridge_mult=0.01)").encode()
    assert (tmp_path / "model.py").read_bytes() == sources["model.py"].replace("0.2", "0.4").encode()
    assert (tmp_path / "score.py").read_bytes() == sources["score.py"].encode()
    recipe.stage_views[0].source_files.append("model.py")
    with pytest.raises(ValueError, match="outside data preparation"):
        validate_axes(recipe, axes, {"prepare"})


def test_study_admission_exposes_scoring_paths_as_strict_records():
    from loop_research.native_study import StudyDraft
    from loop_research.workers import response_schema

    schema = response_schema(StudyDraft)
    binding = schema["properties"]["score_input_bindings"]
    assert binding["type"] == "array"
    fields = schema["$defs"]["ScoreInputBinding"]
    assert set(fields["required"]) == {"logical_path", "source_path"}
    assert fields["additionalProperties"] is False


def test_candidate_admission_supplies_verified_source_and_stage_proof(tmp_path):
    from loop_research.native_study import admission_sources, baseline_receipt_summary
    from loop_research.storage import Evidence

    body = b"#" * 9_000 + b"\nresult = 'complete'\n"
    path = tmp_path / "score.py"
    path.write_bytes(body)
    cards = {"score.py": {"sha256": digest(body), "excerpt": "#"}}
    recipe = NativeRecipe(stages=[], source_files=["score.py"], seed_paths=[], attestations=[],
                          readonly_directories=[], metrics_file=None, metrics=[], questions=[])
    sources, coverage = admission_sources(tmp_path, recipe, cards)
    assert sources["score.py"] == body.decode()
    assert coverage["score.py"]["complete"]
    path.write_bytes(body + b"unexpected")
    with pytest.raises(ValueError, match="Measured admission source changed"):
        admission_sources(tmp_path, recipe, cards)

    store = Evidence(tmp_path / "proofs")
    record = {"stage": "score", "status": "passed", "files": {"metrics.json": digest(b"result")}}
    key = store.put("native_stage_receipt", {**record, "workspace": "/private"})
    measured = store.put("measured_native_baseline", {"stages": [key], "metrics": {"loss": 1}}, [key])
    assert baseline_receipt_summary(store, measured) == {
        "evidence": measured, "stages": [{**record, "evidence": key}], "metrics": {"loss": 1},
    }


def test_admission_reuses_exact_baseline_when_only_manifest_ancestry_changes(tmp_path):
    from loop_research.native_study import record_baseline_revalidation
    from loop_research.storage import Evidence

    store = Evidence(tmp_path)
    history = store.put("native_adaptation_draft", {"revision": 1})
    plan = store.put("onboarding_plan", {"plan": "frozen"})
    stage = store.put("native_stage_receipt", {"stage": "score", "files": {"scores.csv": "original"}})

    def certificate(*, parents=(), source="original", input_sha="original", receipt=stage, loss=2, plan_id=plan):
        manifest = store.put("onboarding_sources", {"cards": [{"path": "score.py", "sha256": source}]}, list(parents))
        recipe = store.put("native_baseline_recipe", {
            "recipe": {"scorer": "score.py"}, "input_sha256": {"holdout.csv": input_sha},
        }, [manifest, plan_id])
        return store.put("measured_native_baseline", {
            "recipe": recipe, "metrics": {"loss": loss}, "stages": [receipt],
        }, [recipe, receipt])

    admitted = certificate(parents=[history])
    measured = certificate()
    proof = store.get(record_baseline_revalidation(store, admitted, measured))
    assert admitted != measured
    assert proof == {"kind": "native_baseline_revalidation",
                     "content": {"admitted": admitted, "measured": measured}, "parents": [admitted, measured]}
    other_plan = store.put("onboarding_plan", {"plan": "different"})
    other_stage = store.put("native_stage_receipt", {"stage": "score", "files": {"scores.csv": "different"}})
    for changed in (certificate(source="changed"), certificate(input_sha="changed"),
                    certificate(receipt=other_stage), certificate(loss=1), certificate(plan_id=other_plan)):
        with pytest.raises(ValueError, match="different measured baseline"):
            record_baseline_revalidation(store, admitted, changed)


def test_measured_priority_conflict_keeps_all_frozen_limits():
    from types import SimpleNamespace

    from loop_research.engine import Research
    from loop_research.models import MetricPolicy

    policy = MetricPolicy.model_validate({
        "targets": [{"name": "mae", "direction": "min", "threshold": 270},
                    {"name": "rmse", "direction": "min", "threshold": 545}],
        "hard_limits": [{"name": "rmse", "direction": "min", "threshold": 548}],
    })
    engine = Research.__new__(Research)
    engine.project = SimpleNamespace(metric_policy=policy, metrics=policy.measured_metrics())
    baseline = {"mae": 279, "rmse": 544}
    feasible = {"mae": 269, "rmse": 546}
    ineligible = {"mae": 267, "rmse": 550}
    assert engine.eligible_improvement(feasible, baseline)
    assert engine.metric_outcome(feasible, baseline)["worse"] == ["rmse"]
    assert not engine.targets_met(feasible)
    assert not engine.eligible_improvement(ineligible, baseline)
    assert engine.metric_outcome(ineligible, baseline)["failed_limits"] == ["rmse"]
    assert engine.targets_met({"mae": 268, "rmse": 544})


def test_native_replay_accepts_only_one_float_step_with_same_decision():
    from math import inf, nextafter
    from types import SimpleNamespace

    from loop_research.engine import Research
    from loop_research.models import Metric, MetricPolicy

    engine = Research.__new__(Research)
    engine.command = SimpleNamespace(command=["isolated-native"])
    policy = MetricPolicy(targets=[Metric(name="loss", direction="min", threshold=2.5)])
    engine.project = SimpleNamespace(metric_policy=policy, metrics=policy.measured_metrics())
    state = SimpleNamespace(best_metrics={"loss": 3.0}, active_tasks=[])
    original = {"status": "succeeded", "metrics": {"loss": 2.0}, "train_indices": ["same"]}
    replay = {**original, "metrics": {"loss": nextafter(2.0, inf)}}
    chosen = SimpleNamespace(id="candidate")
    assert engine.replay_comparison(state, chosen, original, replay)[0]
    assert not engine.replay_comparison(state, chosen, original, {**replay, "train_indices": ["changed"]})[0]
    assert not engine.replay_comparison(state, chosen, original, {**replay, "metrics": {"loss": 2.00001}})[0]
    state.best_metrics = original["metrics"]
    assert not engine.replay_comparison(state, chosen, original, replay)[0]
    state.best_metrics = {"loss": 3.0}
    policy.targets[0].threshold = 2.0
    assert not engine.replay_comparison(state, chosen, original, replay)[0]


def test_native_multifile_candidate_uses_exact_source_scope_and_independent_score(tmp_path):
    require_bubblewrap()
    bundle, run = tmp_path / "bundle", tmp_path / "run"
    (bundle / "native").mkdir(parents=True)
    run.mkdir()
    sources = {
        "train.py": "from pathlib import Path\nfrom learner import transform\n"
                    "assert not Path('score.py').exists() and not Path('label.txt').exists()\n"
                    "Path('predictions').write_text(str(transform(int(Path('data.txt').read_text()))))\n",
        "learner.py": "from features import convert\ndef transform(value):\n    return convert(value) + 1\n",
        "features.py": "def convert(value):\n    return value * 2\n",
        "score.py": "from pathlib import Path\nimport json\n"
                    "assert not Path('learner.py').exists() and not Path('features.py').exists()\n"
                    "assert not Path('label.txt').exists()\n"
                    "loss = abs(int(Path('data.txt').read_text()) - int(Path('predictions').read_text()))\n"
                    "Path('metrics.json').write_text(json.dumps({'loss':loss}))\n",
    }
    for name, body in sources.items():
        (bundle / "native" / name).write_text(body)
    files = ["features.py", "learner.py"]
    recipe = NativeRecipe(
        stages=[{"name": "train", "script": "train.py", "outputs": ["predictions"]},
                {"name": "score", "script": "score.py", "outputs": ["metrics.json"]}],
        stage_views=[
            {"stage": "train", "source_files": ["train.py", *files], "input_files": ["data.txt"],
             "previous_outputs": []},
            {"stage": "score", "source_files": ["score.py"], "input_files": ["data.txt"],
             "previous_outputs": ["predictions"]},
        ],
        source_files=list(sources), input_files=["data.txt", "label.txt"],
        score_input_bindings=[{"logical_path": "data.txt", "source_path": "label.txt"}],
        seed_paths=[], attestations=[],
        readonly_directories=[], metrics_file="metrics.json", metrics=[{"name": "loss", "keys": ["loss"]}],
        questions=[],
    )
    with pytest.raises(ValueError, match="invisible to the scorer"):
        validate_mutable_sources(recipe, [*files, "score.py"], ["train"])
    data = {"validation": {"data.txt": b"4", "label.txt": b"14"},
            "test": {"data.txt": b"4", "label.txt": b"16"}}
    for split, artifacts in data.items():
        folder = bundle / "views" / split
        folder.mkdir(parents=True)
        for name, value in artifacts.items():
            (folder / name).write_bytes(value)
    (bundle / "split.json").write_bytes(canonical({"train": ["t"], "validation": ["v"], "test": ["h"]}))
    (bundle / "native-study.json").write_bytes(canonical({
        "recipe": recipe.model_dump(), "axes": [], "mutable_sources": files, "mutable_stages": ["train"],
        "view_sha256": {split: {name: digest(value) for name, value in artifacts.items()}
                        for split, artifacts in data.items()},
    }))
    from loop_research.native_study import commit_bundle

    revision = commit_bundle(bundle)
    config = CommandExperiment(repository=str(bundle), revision=revision, command=["isolated-native"],
                               split_manifest="split.json", baseline={}, choices={}, mutable_sources=files,
                               min_structural_files=2, bridge_interpreter=sys.executable)
    Project(name="scope", objective="Compare a paired structural intervention", knowledge="The score is independently frozen.",
            archive_uri=str(run), metrics=[{"name": "loss", "direction": "min", "threshold": 0}], max_experiments=1,
            branches=1, concurrency=1, reversible=True, affordable=True, confirmed_knowledge=True,
            command_experiment=config)
    source = prepare(run, config)
    baseline = frozen_baseline_code(run, config)
    parameters, modules = candidate_parts(baseline, config)
    edited = {**modules, "features.py": "def convert(value):\n    return value * 3\n",
              "learner.py": "from features import convert\ndef transform(value):\n    return convert(value) + 2\n"}
    from loop_research.engine import Research
    engine = Research.__new__(Research)
    engine.command = config
    with pytest.raises(ValueError, match="fewer than the required"):
        engine.intervention(baseline, candidate_code(parameters, {**modules, "features.py": edited["features.py"]}))
    assert engine.intervention(baseline, candidate_code(parameters, edited)) == "model.structure"
    with pytest.raises(ValueError, match="exact admitted source files"):
        candidate_parts(candidate_code(parameters, {**edited, "score.py": "pass"}), config)
    original = measure_native(run, config, source, baseline, 9, "validation", run / "baseline")
    candidate = measure_native(run, config, source, candidate_code(parameters, edited), 9, "validation", run / "candidate")
    assert original["metrics"] == {"loss": 5}
    assert candidate["metrics"] == {"loss": 0}
    assert candidate["data_sha256"] == original["data_sha256"]
    assert all((bundle / "native" / name).read_text() == text for name, text in sources.items())


def test_training_scope_is_measured_with_original_scorer_and_never_touches_source(tmp_path):
    from loop_research.native_training_scope import prepare_training_scope
    from loop_research.onboard_replay import runtime_interpreter
    from loop_research.storage import Evidence

    require_bubblewrap()
    project, root = tmp_path / "project", tmp_path / "run"
    project.mkdir()
    sources = {
        "train.py": "from pathlib import Path\nPath('pred').write_text('3')\n",
        "score.py": "import json\nfrom pathlib import Path\n"
                    "assert not Path('loop-training-scope').exists()\n"
                    "Path('metrics.json').write_text(json.dumps({'loss': int(Path('pred').read_text())}))\n",
    }
    cards = {}
    for name, code in sources.items():
        (project / name).write_text(code)
        cards[name] = {"path": name, "excerpt": code, "sha256": digest(code.encode())}
    recipe = NativeRecipe(
        stages=[{"name": "train", "script": "train.py", "outputs": ["pred"]},
                {"name": "score", "script": "score.py", "outputs": ["metrics.json"]}],
        source_files=list(sources), seed_paths=[], attestations=[], readonly_directories=[], questions=[],
        stage_views=[{"stage": "train", "source_files": ["train.py"], "input_files": [], "previous_outputs": []},
                     {"stage": "score", "source_files": ["score.py"], "input_files": [], "previous_outputs": ["pred"]}],
        working_directories=[{"stage": "train", "path": "."}, {"stage": "score", "path": "."}],
        metrics_file="metrics.json", metrics=[{"name": "loss", "keys": ["loss"]}],
    )
    runner = runtime_interpreter(Path(sys.executable))
    _, actual = run_recipe(project, root / "old", runner, None, recipe, cards)
    store = Evidence(root)
    recipe_ref = store.put("native_baseline_recipe", {"recipe": recipe.model_dump()})
    baseline = store.put("measured_native_baseline", {"metrics": actual, "recipe": recipe_ref}, [recipe_ref])
    draft = {
        "stage": "train", "blockers": [], "explanation": "Separate two existing training contributions.",
        "driver": {"path": "loop-training-scope/train.py", "content":
                   "from pathlib import Path\nfrom first import first\nfrom second import second\n"
                   "Path('pred').write_text(str(first()+second()))\n"},
        "modules": [{"path": "loop-training-scope/first.py", "content": "def first(): return 1\n"},
                    {"path": "loop-training-scope/second.py", "content": "def second(): return 2\n"}],
    }

    class Worker:
        def run(self, role, payload, output):
            result = draft if role == "native-training-scope" else {"accepted": True, "findings": []}
            return output.model_validate(result), {"role": role}

    workspace, scoped, _, _, proof = prepare_training_scope(
        root, root / "candidate", project, runner, recipe, cards, baseline, Worker(), 2)
    assert store.get(proof)["content"]["metrics"] == actual == {"loss": 3}
    assert scoped.stages[-1] == recipe.stages[-1]
    assert (workspace / "score.py").read_text() == sources["score.py"]
    assert not (project / "loop-training-scope").exists()


def test_stage_views_hide_frozen_files_and_preserve_original_outputs_on_resume(tmp_path):
    require_bubblewrap()
    project = tmp_path / "project"
    project.mkdir()
    sources = {
        "train.py": """from pathlib import Path
assert not Path('score.py').exists()
assert not Path('labels.txt').exists()
Path('model').write_text(Path('train.txt').read_text())
Path('unlisted').write_text('should not reach another stage')
""",
        "predict.py": """from pathlib import Path
assert not Path('train.py').exists()
assert not Path('train.txt').exists()
assert not Path('labels.txt').exists()
assert not Path('unlisted').exists()
Path('predictions').write_text(Path('model').read_text())
""",
        "score.py": """import json
from pathlib import Path
assert not Path('train.py').exists()
assert not Path('predict.py').exists()
assert not Path('train.txt').exists()
assert not Path('unlisted').exists()
assert Path('model').read_text() == Path('predictions').read_text()
correct = Path('labels.txt').read_text() == Path('predictions').read_text()
Path('metrics.json').write_text(json.dumps({'accuracy': int(correct)}))
""",
    }
    for name, data in {**sources, "train.txt": "5", "labels.txt": "5"}.items():
        (project / name).write_text(data)
    cards = {name: {"sha256": digest(data.encode())} for name, data in sources.items()}
    recipe = NativeRecipe(
        stages=[{"name": "train", "script": "train.py", "outputs": ["model"]},
                {"name": "predict", "script": "predict.py", "outputs": ["predictions"]},
                {"name": "score", "script": "score.py", "outputs": ["metrics.json"]}],
        source_files=list(sources), input_files=["train.txt", "labels.txt"],
        stage_views=[
            {"stage": "train", "source_files": ["train.py"], "input_files": ["train.txt"], "previous_outputs": []},
            {"stage": "predict", "source_files": ["predict.py"], "input_files": [], "previous_outputs": ["model"]},
            {"stage": "score", "source_files": ["score.py"], "input_files": ["labels.txt"],
             "previous_outputs": ["model", "predictions"]},
        ],
        seed_paths=[], attestations=[], readonly_directories=[], metrics_file="metrics.json",
        metrics=[{"name": "accuracy", "keys": ["accuracy"]}], questions=[],
    )
    plan = {"baseline_entrypoints": ["train.py", "predict.py"], "independent_evaluation_entrypoints": ["score.py"]}
    validate_recipe(recipe, project, None, plan, cards)
    args = (project, tmp_path / "run", Path(sys.executable).resolve(), None, recipe, cards)
    receipts, result = run_recipe(*args)
    assert result == {"accuracy": 1}
    assert run_recipe(*args) == (receipts, result)
    assert not (project / "model").exists()
    recipe.stage_views[0].previous_outputs.append("predictions")
    with pytest.raises(ValueError, match="undeclared files or earlier outputs"):
        validate_recipe(recipe, project, None, plan, cards)


def test_curation_keeps_large_native_measurements_inside_evidence():
    from loop_research.engine import curation_observation

    indices = [f"original-row:{i}" for i in range(53_940)]
    measurement = {"metrics": {"mae": 277.2}, "split": "validation", "train_indices": indices}
    observation = curation_observation("measurement", {"kind": "training_result", "content": measurement})
    assert observation["content"]["metrics"] == measurement["metrics"]
    assert observation["content"]["arrays"]["train_indices"] == {
        "count": len(indices), "sha256": digest(canonical(indices)),
    }
    assert len(canonical(observation)) < 1000
    assert measurement["train_indices"] == indices
