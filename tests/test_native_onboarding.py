"""Regression coverage for discovered entrypoints and real isolated stage handoff."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner
from test_study_intake import existing_experiment

from loop_research.cli import app
from loop_research.native_adaptation import (AdaptationDraft, AdaptationRevision, apply_adaptation_revision,
                                             rejected_adapter_to_revise)
from loop_research.onboard_native import (FrozenInputChanged, NativeRecipe, failure_context, qualify, read_metrics,
                                           run_recipe)
from loop_research.onboarding import OnboardingPlan, catalogue, experiment_signals, source_cards, supporting_sources
from loop_research.sealed_stage import file_hash
from loop_research.storage import Evidence
from loop_research.workers import WorkerFailure, codex_executable, response_schema


def test_entrypoint_and_source_choices_use_different_enums():
    props = response_schema(OnboardingPlan, source_paths=["train.py", "config.json"],
                            entrypoint_paths=["train.py"])["properties"]
    assert props["evidence_paths"]["items"]["enum"] == ["train.py", "config.json"]
    assert props["baseline_entrypoints"]["items"]["enum"] == ["train.py"]
    assert props["independent_evaluation_entrypoints"]["items"]["enum"] == ["train.py"]


def test_discovery_includes_experiments_without_main_guard(tmp_path):
    demos = tmp_path / "source" / "examples"
    demos.mkdir(parents=True)
    raw = b'"""Multi-objective accuracy and cost study."""\n\ndef objective(trial):\n    return evaluate(trial)\n\nstudy.optimize(objective)\n'
    (demos / "choice_tradeoffs.py").write_bytes(raw)
    (demos / "mentions_experiments.py").write_text('"""Call study.optimize(objective) to start."""')
    assert catalogue(demos.parent, tmp_path / "run") == ["examples/choice_tradeoffs.py"]
    assert experiment_signals(raw) == {"training_calls": ["optimize"],
                                       "evaluation_symbols": ["evaluate", "objective"],
                                       "description": "Multi-objective accuracy and cost study."}


def test_adapter_revision_recovers_review_after_a_timeout(tmp_path):
    store = Evidence(tmp_path)
    original = AdaptationDraft(recipe=None, generated=[
        {"path": "loop-adapters/a.py", "content": "verified = False"},
        {"path": "loop-adapters/b.py", "content": "frozen_scorer = True"},
    ], reference_metrics=[], holdout_inputs=["sealed/test.csv"], blockers=[], limitations=[]).model_dump()
    base = store.put("native_adaptation_draft", {"draft": original})
    review = store.put("native_adaptation_review", {"review": {"accepted": False, "findings": ["Check inputs"]}}, [base])
    rejected = store.put("native_adaptation_failed", {"reason": "review rejected"}, [base, review])
    timed_out = store.put("native_adaptation_failed", {"reason": "timeout"}, [rejected])
    assert rejected_adapter_to_revise(store, timed_out) == {"base": base, "review": review}
    rebuild = store.put("native_adaptation_failed", {
        "reason": "Adapter requires full revision: distrust an obsolete generated file",
    }, [rejected, base, review])
    rebuild_timeout = store.put("native_adaptation_failed", {"reason": "timeout"}, [rebuild])
    assert rejected_adapter_to_revise(store, rebuild_timeout) == {
        "base": base, "review": review, "technical_failure": rebuild,
    }
    options = AdaptationRevision(replacement_sources=[{"path": "loop-adapters/a.py", "content": "verified = True"}],
                                 recipe=None, limitations=None, rebuild_reason=None)
    repaired = apply_adaptation_revision(original, options).model_dump()
    assert repaired["holdout_inputs"] == original["holdout_inputs"]
    assert repaired["generated"][1] == original["generated"][1]
    assert repaired["generated"][0]["content"] == "verified = True"
    remove_obsolete = options.model_copy(update={"removed_sources": ["loop-adapters/b.py"]})
    assert len(apply_adaptation_revision(original, remove_obsolete).generated) == 1
    attempt_upstream = options.model_copy(update={"removed_sources": ["upstream/score.py"]})
    with pytest.raises(ValueError, match="existing generated files"):
        apply_adaptation_revision(original, attempt_upstream)


def require_bubblewrap():
    if not sys.platform.startswith("linux"):
        pytest.skip("Native qualification uses Linux bubblewrap")
    if not shutil.which("bwrap"):
        try:
            binary = Path(codex_executable()).parent.parent / "codex-resources" / "bwrap"
        except (OSError, WorkerFailure):
            pytest.skip("bubblewrap unavailable")
        if not os.access(binary, os.X_OK):
            pytest.skip("bubblewrap unavailable")


def test_native_onboarding_repairs_and_resumes_real_scripts(tmp_path):
    require_bubblewrap()
    source = tmp_path / "original"
    source.mkdir()
    inputs = tmp_path / "existing-artifacts"
    inputs.mkdir()
    root = tmp_path / "run"
    (source / "calibration.json").write_text('{"calibration":3}')
    (source / "config.json").write_text('{"model":"fixture"}')
    (source / "PROTOCOL.md").write_text("Run train_probe.py then evaluate.py. Train and score separately.")
    (source / "custom_entry.py").write_text("if __name__ == '__main__':\n    pass\n")
    (source / "train_probe.py").write_text("""
import json
from pathlib import Path
raw = json.loads(Path("calibration.json").read_text())
Path("fit.json").write_text(json.dumps({"fit": raw["calibration"] + 2}))
Path("model").mkdir()
Path("model/artifact.txt").write_text("fit proof")
""")
    (source / "evaluate.py").write_text("""
import json
from pathlib import Path
import sys
if sys.argv[1:] != ["--score"]:
    raise SystemExit(2)
fit = json.loads(Path("fit.json").read_text())["fit"]
calibration = json.loads(Path("calibration.json").read_text())["calibration"]
Path("result.json").write_text(json.dumps({"metrics": {"quality": fit - calibration}}))
""")
    (inputs / "seal.json").write_text(json.dumps({
        "source_hashes": {str(source / "calibration.json"): file_hash(source / "calibration.json")},
    }))

    class Worker:
        calls = []

        def run(self, role, payload, response):
            self.calls.append(role)
            if role == "onboard-discover":
                assert "train_probe.py" in payload["catalogue"]
                assert "custom_entry.py" in payload["catalogue"]
                result = {"paths": ["PROTOCOL.md", "config.json", "evaluate.py"], "rationale": "fixture"}
            elif role == "onboard-design":
                assert "config.json" not in payload["entrypoint_paths"]
                valid = self.calls.count(role) == 2
                if valid:
                    assert "config.json" in payload["previous_rejection"]["reason"]
                result = {"project": "fixture", "research_question": "Improve the independent reported quality",
                          "evidence_paths": ["train_probe.py", "evaluate.py", "PROTOCOL.md", "config.json"],
                          "dataset_and_split": "fixture", "current_baseline": "fixture",
                          "baseline_entrypoints": ["train_probe.py" if valid else "config.json"],
                          "independent_evaluation_entrypoints": ["evaluate.py"], "preserved_capabilities": [],
                          "adaptation_steps": [], "blocking_questions": [], "first_validation": "fixture"}
            elif role == "onboard-review":
                result = {"supported": True, "findings": [], "proven_preservation": [], "missing_prerequisites": []}
            elif role == "onboard-native-recipe":
                valid = self.calls.count(role) == 2
                if valid:
                    assert payload["previous_failure"]["stage"] == "score"
                    assert "exit" in payload["previous_failure"]["error"].lower()
                result = {"stages": [
                            {"script": "train_probe.py", "name": "train", "arguments": [], "outputs": ["fit.json", "model"]},
                            {"script": "evaluate.py", "name": "score",
                             "arguments": ["--score" if valid else "--wrong"], "outputs": ["result.json"]}],
                          "source_files": ["train_probe.py", "evaluate.py", "PROTOCOL.md"],
                          "seed_paths": ["seal.json"], "attestations": ["seal.json"], "readonly_directories": [],
                          "metrics_file": "result.json", "metrics": [{"name": "quality", "keys": ["metrics", "quality"]}],
                          "questions": []}
            else:
                assert role == "onboard-native-review"
                result = {"accepted": True, "findings": []}
            return response.model_validate(result), {"role": role}

    worker = Worker()
    objective = "Measure the original fixture pipeline."
    pending = qualify(source, root, objective, artifacts=inputs, worker=worker)
    assert pending["phase"] == "awaiting_input" and pending["issues"][0]["field"] == "interpreter"
    complete = qualify(source, root, objective, interpreter=Path(sys.executable),
                       artifacts=inputs, worker=worker)
    assert complete["phase"] == "baseline_verified", complete
    assert complete["metrics"] == {"quality": 2}
    assert len(list((root / "native-stages").glob("*/train/*/receipt.json"))) == 1
    assert len(list((root / "native-stages").glob("*/score/*/receipt.json"))) == 2
    assert not (source / "fit.json").exists() and not (source / "result.json").exists()
    status = CliRunner().invoke(app, ["status", str(root)])
    assert status.exit_code == 0 and json.loads(status.stdout)["phase"] == "baseline_verified"

    calls = list(worker.calls)
    completed_pointer = next((root / "native-stages").glob("*/score.json"))
    completed_pointer.unlink()
    complete["phase"] = "execute"
    complete["retained"] = [key for key in complete["retained"]
                            if Evidence(root).get(key)["content"]["stage"] == "train"]
    (root / "native-onboarding.json").write_text(json.dumps(complete))
    assert qualify(source, root, objective, worker=worker)["phase"] == "baseline_verified"
    assert len(list((root / "native-stages").glob("*/score/*/receipt.json"))) == 2
    assert worker.calls == calls
    fitting = next((root / "native-stages").glob("*/train/*/replica/fit.json"))
    fitting.write_text("{}")
    assert qualify(source, root, objective, worker=worker)["phase"] == "integrity_error"
    assert worker.calls == calls


def test_native_reference_reads_the_declared_fold_from_a_numeric_artifact(tmp_path):
    from decimal import Decimal

    from loop_research.native_adaptation import ReferenceMetric, compare_reference, reference_values

    (tmp_path / "result.json").write_text('{"validation":{"AUC":[0.91,0.88]},"test":{"AUC":[0.97,0.94]}}')
    metric = ReferenceMetric(name="auc", artifact="result.json", json_path=["validation", "AUC", "0"],
                             selection="last", expected_observations=1)
    original = reference_values(tmp_path / "unused.stdout", [metric], tmp_path)
    assert original == {"auc": Decimal("0.91")}
    scoring = NativeRecipe(stages=[], source_files=[], seed_paths=[], attestations=[], readonly_directories=[],
                           metrics_file="result.json", metrics=[{"name": "auc", "keys": metric.json_path}],
                           questions=[])
    assert read_metrics(tmp_path, tmp_path, scoring) == {"auc": 0.91}
    scoring.metrics[0].keys[-1] = "-1"
    with pytest.raises(ValueError, match="did not return metric auc"):
        read_metrics(tmp_path, tmp_path, scoring)
    compare_reference(original, {"auc": 0.911})
    with pytest.raises(ValueError, match="differs from the native development metric"):
        compare_reference(original, {"auc": 0.97})


def test_adapter_context_keeps_entrypoints_and_reviewed_dependencies_before_optional_sources():
    from loop_research.native_adaptation import SOURCE_CONTEXT_LIMIT, adaptation_context

    texts = {f"optional-{index}.py": "#" * 8_000 for index in range(150)}
    texts.update({"baseline.py": "def main(): pass\n", "scorer.py": "def score(): return 1\n"})
    plan = {"baseline_entrypoints": ["baseline.py"], "evidence_paths": []}
    context = adaptation_context(texts, plan, ["scorer.py"])
    assert context["sources"]["baseline.py"] == texts["baseline.py"]
    assert context["sources"]["scorer.py"] == texts["scorer.py"]
    assert sum(map(len, context["sources"].values())) <= SOURCE_CONTEXT_LIMIT
    assert not context["source_coverage"]["optional-149.py"]["complete"]


def test_coupled_entrypoint_qualifies_generated_stages_against_original_output(tmp_path):
    from loop_research.native_adaptation import qualify_adaptation
    from loop_research.onboard_replay import runtime_interpreter

    require_bubblewrap()
    project, run = tmp_path / "project", tmp_path / "run"
    project.mkdir()
    sources = {"main.py": "from pathlib import Path\nassert Path('test.txt').exists()\n"
                          "Path('upstream').write_text('checkpoint')\nprint('development loss 1.20')\n"}
    for name, text in {**sources, "train.txt": "four", "valid.txt": "example", "test.txt": "secret"}.items():
        (project / name).write_text(text)
    reference = {"name": "reference", "script": "main.py", "outputs": ["upstream"]}
    scripts = {
        "loop-adapters/seal.py": "import json\nfrom pathlib import Path\n"
                                 "Path('dimension.json').write_text(json.dumps({'dimension': len(Path('test.txt').read_text())}))\n",
        "loop-adapters/train.py": "import json\nfrom pathlib import Path\n"
                                  "assert not Path('test.txt').exists() and not Path('upstream').exists()\n"
                                  "assert json.loads(Path('dimension.json').read_text()) == {'dimension': 6}\n"
                                  "Path('weights').write_text(Path('train.txt').read_text())\n",
        "loop-adapters/score.py": "from pathlib import Path\nassert not Path('test.txt').exists()\n"
                                  "assert not Path('train.txt').exists() and Path('valid.txt').exists()\n"
                                  "assert Path('weights').read_text() == 'four'\n"
                                  "Path('metrics.json').write_text('{\"loss\": 1.203}')\n",
    }
    steps = [reference, {"name": "seal-development-metadata", "script": "loop-adapters/seal.py",
                         "outputs": ["dimension.json"]},
             {"name": "train", "script": "loop-adapters/train.py", "outputs": ["weights"]},
             {"name": "score", "script": "loop-adapters/score.py", "outputs": ["metrics.json"]}]
    views = [dict(stage=step["name"], source_files=[step["script"]], input_files=inputs, previous_outputs=outputs)
             for step, inputs, outputs in zip(steps, [["test.txt"], ["test.txt"], ["train.txt"], ["valid.txt"]],
                                             [[], [], ["dimension.json"], ["weights"]], strict=True)]
    draft = {
        "recipe": {"stages": steps, "source_files": [*sources, *scripts], "seed_paths": [], "attestations": [],
                   "readonly_directories": [], "input_files": ["train.txt", "valid.txt", "test.txt"],
                   "working_directories": [{"stage": step["name"], "path": "."} for step in steps],
                   "metrics_file": "metrics.json", "metrics": [{"name": "loss", "keys": ["loss"]}],
                   "questions": [], "stage_views": views},
        "generated": [{"path": name, "content": text} for name, text in scripts.items()],
        "reference_metrics": [{"name": "loss", "line_marker": "development", "label": "loss", "selection": "min"}],
        "holdout_inputs": ["test.txt"], "development_metadata": {"output": "dimension.json", "integer_fields": ["dimension"]},
        "blockers": [], "limitations": [],
    }
    store = Evidence(run)
    plan = {"baseline_entrypoints": ["main.py"], "independent_evaluation_entrypoints": []}
    plan_ref = store.put("onboarding_plan", {"plan": plan})
    source = {"plan": plan_ref, "review": store.put("onboarding_review", {}, [plan_ref]),
              "sources": store.put("onboarding_sources", {"cards": source_cards(project, list(sources), set(sources))})}

    class Worker:
        calls = []

        def run(self, role, payload, response):
            self.calls.append(role)
            assert payload["existing_file_inventory"]["project"] == ["main.py", "test.txt", "train.txt", "valid.txt"]
            result = draft if role == "onboard-coupled-adapter" else {"accepted": True, "findings": []}
            return response.model_validate(result), {"role": role}

    worker = Worker()
    args = (project, run, "Measure", source, runtime_interpreter(Path(sys.executable)), None, plan, worker)
    certified = qualify_adaptation(*args)
    assert certified["phase"] == "baseline_verified", certified
    assert certified["metrics"] == {"loss": 1.203}
    reusable = store.get(certified["recipe"])["content"]
    assert set(reusable["input_sha256"]) == {"train.txt", "valid.txt", "dimension.json"}
    assert not (project / "loop-adapters").exists()
    assert qualify_adaptation(*args) == certified
    assert worker.calls == ["onboard-coupled-adapter", "onboard-coupled-review"]


@pytest.mark.parametrize("as_callable", [False, True], ids=["original-cli", "native-return"])
def test_coupled_entrypoint_prepares_raw_data_then_seeds_the_original_script(tmp_path, as_callable):
    from loop_research.native_adaptation import qualify_adaptation
    from loop_research.onboard_replay import runtime_interpreter

    require_bubblewrap()
    project, run, data = (tmp_path / name for name in ("project", "run", "data"))
    (project / "pipeline").mkdir(parents=True)
    data.mkdir()
    (data / "raw.txt").write_text("train,validation,hidden")
    entrypoint = "pipeline/main.py"
    original = (
        "import json, random, sys\nfrom pathlib import Path\n"
        "assert sys.argv[1:] == ['native']\n"
        "assert Path('../reference.txt').read_text() == 'train,validation'\n"
        "assert not Path('../raw.txt').exists() and not Path('../private.json').exists()\n"
        "Path('../native.json').write_text(json.dumps({'val': [random.random()]}))\n"
    )
    if as_callable:
        original = (
            "import random\nfrom pathlib import Path\n"
            "DATA = 'https://example.invalid/reference.txt'\n"
            "def experiment(mode):\n"
            "    assert mode == 'native' and Path(DATA).read_text() == 'train,validation'\n"
            "    assert not Path('../raw.txt').exists() and not Path('../private.json').exists()\n"
            "    return [{'val': random.random()}]\n"
            "if __name__ == '__main__':\n"
            "    raise RuntimeError('the reference calls the native function')\n"
        )
    (project / entrypoint).write_text(original)
    scripts = {
        "loop-adapters/prepare.py": "from pathlib import Path\n"
                                    "train,val,test = Path('raw.txt').read_text().split(',')\n"
                                    "Path('reference.txt').write_text(train + ',' + val)\n"
                                    "Path('train.txt').write_text(train)\n"
                                    "Path('validation.txt').write_text(val)\n"
                                    "Path('private.json').write_text('{\"train\":[1],\"validation\":[2],\"test\":[3]}')\n",
        "loop-adapters/train.py": "import random\nfrom pathlib import Path\n"
                                  "assert Path('train.txt').read_text() == 'train'\n"
                                  "assert not any(Path(name).exists() for name in "
                                  "['raw.txt', 'private.json', 'reference.txt', 'validation.txt'])\n"
                                  "Path('weights').write_text(str(random.Random(23).random()))\n",
        "loop-adapters/score.py": "import json\nfrom pathlib import Path\n"
                                  "assert Path('validation.txt').read_text() == 'validation'\n"
                                  "assert not any(Path(name).exists() for name in "
                                  "['raw.txt', 'private.json', 'reference.txt', 'train.txt', 'native.json', 'reference-return.json'])\n"
                                  "Path('metrics.json').write_text(json.dumps({'score': float(Path('weights').read_text())}))\n",
    }
    prepared_files = ["reference.txt", "train.txt", "validation.txt", "private.json"]
    steps = [
        {"name": "prepare-native-inputs", "script": "loop-adapters/prepare.py", "outputs": prepared_files},
        {"name": "reference", "script": entrypoint, "arguments": ["native"], "outputs": ["native.json"]},
        {"name": "train", "script": "loop-adapters/train.py", "outputs": ["weights"]},
        {"name": "score", "script": "loop-adapters/score.py", "outputs": ["metrics.json"]},
    ]
    views = [dict(stage=step["name"], source_files=[step["script"]], input_files=inputs, previous_outputs=outputs)
             for step, inputs, outputs in zip(steps, [["raw.txt"], [], [], []],
                 [[], ["reference.txt"], ["train.txt"], ["validation.txt", "weights"]], strict=True)]
    reference_call = None
    reference_metric = {"name": "score", "artifact": "native.json", "json_path": ["val", "0"], "selection": "last"}
    if as_callable:
        steps[1].update(arguments=[], outputs=["reference-return.json"])
        reference_metric.update(artifact="reference-return.json", json_path=["0", "val"])
        reference_call = {"function": "experiment", "arguments": [{"name": "mode", "value": "native"}],
                          "offline_inputs": [{"global_name": "DATA", "path": "reference.txt"}], "result_columns": []}
    draft = {
        "recipe": {"stages": steps, "source_files": [entrypoint, *scripts], "seed_paths": [], "attestations": [],
                   "readonly_directories": [], "input_files": ["raw.txt"],
                   "working_directories": [{"stage": step["name"], "path": "."} for step in steps if step["name"] != "reference"],
                   "metrics_file": "metrics.json", "metrics": [{"name": "score", "keys": ["score"]}],
                   "questions": [], "stage_views": views},
        "generated": [{"path": name, "content": text} for name, text in scripts.items()],
        "reference_metrics": [reference_metric], "reference_call": reference_call,
        "holdout_inputs": ["raw.txt"], "reference_seed": {"seed": 23, "libraries": []},
        "trusted_preparation": {"split_manifest": "private.json", "reference_files": ["reference.txt"],
                                "development_files": ["train.txt", "validation.txt"]},
        "blockers": [], "limitations": [],
    }
    store = Evidence(run)
    plan = {"baseline_entrypoints": [entrypoint], "independent_evaluation_entrypoints": []}
    plan_ref = store.put("onboarding_plan", {"plan": plan})
    source = {"plan": plan_ref, "review": store.put("onboarding_review", {}, [plan_ref]),
              "sources": store.put("onboarding_sources", {"cards": source_cards(project, [entrypoint], {entrypoint})})}

    class Worker:
        def run(self, role, payload, response):
            result = draft if role == "onboard-coupled-adapter" else {"accepted": True, "findings": []}
            return response.model_validate(result), {"role": role}

    certified = qualify_adaptation(project, run, "Measure", source, runtime_interpreter(Path(sys.executable)),
                                   data, plan, Worker())
    assert certified["phase"] == "baseline_verified", certified
    assert [store.get(ref)["content"]["stage"] for ref in certified["retained"]] == [step["name"] for step in steps]
    reusable = store.get(certified["recipe"])["content"]
    assert set(reusable["input_sha256"]) == {"train.txt", "validation.txt"}
    assert [step["name"] for step in reusable["recipe"]["stages"]] == ["train", "score"]
    assert not (project / "loop-adapters").exists()


def test_native_import_closure_includes_script_search_root_for_nested_modules(tmp_path):
    files = {"pipeline/run.py": "from trainers.engine import fit\n", "pipeline/trainers/engine.py": (
        "from helpers.io import save\n"
        "def fit():\n"
        "    import optional_backend\n"
    ), "pipeline/helpers/io.py": "def save(): pass\n", "pipeline/optional_backend.py": ""}
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    sources = supporting_sources(tmp_path, ["pipeline/run.py", "pipeline/trainers/engine.py"],
                                 entrypoints=["pipeline/run.py"], include_lazy=False)
    assert sources == ["pipeline/helpers/io.py"]


def test_native_study_monitor_can_follow_training_past_the_old_poll_cap(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from loop_research import cli

    revisions = iter(range(1, 253))
    reports = []

    class Controller:
        def __init__(self, root):
            assert root == tmp_path

        def tick(self):
            revision = next(revisions)
            phase = "complete" if revision == 252 else "monitor"
            return SimpleNamespace(revision=revision, phase=phase, batch=1, used=int(phase == "complete"))

        def report(self):
            reports.append("complete")

    monkeypatch.setattr(cli, "Research", Controller)
    monkeypatch.setattr(cli.time, "sleep", lambda interval: None)
    result = CliRunner().invoke(app, ["run", str(tmp_path), "--max-ticks", "0"])
    assert result.exit_code == 0 and reports == ["complete"]
    assert "revision=252 phase=complete" in result.stdout


def test_traces_project_support_and_checks_only_new_score_metrics(tmp_path):
    project = tmp_path / "project"
    run = tmp_path / "run"
    artifacts = tmp_path / "artifacts"
    files = {
        "train.py": "from package.calc import fit\nfit()\n",
        "evaluate.py": "from package.calc import evaluate\nevaluate()\n",
        "package/__init__.py": "",
        "package/calc.py": (
            "from pathlib import Path\nfrom .output import write\n"
            "def fit():\n"
            "    multiplier = int(Path('config/pilot.ini').read_text())\n"
            "    value = int(Path('data/frozen.csv').read_text()) * multiplier\n"
            "    Path('prediction.txt').write_text(str(value))\n"
            "    write('99')\n"
            "def evaluate():\n    write(Path('prediction.txt').read_text())\n"
        ),
        "package/output.py": (
            "def write(value):\n"
            "    with open('results.txt', 'a') as report:\n"
            "        report.write('Score header\\nAccuracy - mean: ' + value + '\\n')\n"
        ),
        "config/pilot.ini": "2",
        "README.md": "Run training and independent scoring",
    }
    for name, content in files.items():
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (project / "data").mkdir()
    (project / "data/other.csv").write_text("unrelated data")
    (project / "config/linked.yaml").symlink_to(project / "data/other.csv")
    support = supporting_sources(project, ["train.py", "evaluate.py"])
    assert set(support) == {"package/__init__.py", "package/calc.py", "package/output.py", "config/pilot.ini"}

    class Worker:
        checked_question = False

        def run(self, role, payload, response):
            if role == "onboard-discover":
                result = {"paths": ["README.md", "train.py", "evaluate.py"], "rationale": "fixture"}
            elif role == "onboard-design":
                assert set(support) <= set(payload["source_paths"])
                assert not set(support) & set(payload["entrypoint_paths"])
                result = {"project": "fixture", "research_question": "Measure original pipeline",
                          "evidence_paths": ["train.py", "evaluate.py", "package/calc.py"],
                          "dataset_and_split": "fixture", "current_baseline": "fixture",
                          "baseline_entrypoints": ["train.py"], "independent_evaluation_entrypoints": ["evaluate.py"],
                          "preserved_capabilities": [], "adaptation_steps": [], "blocking_questions": [],
                          "first_validation": "fixture"}
            elif role == "onboard-review":
                result = {"supported": True, "findings": [], "proven_preservation": [], "missing_prerequisites": []}
            elif role == "onboard-native-recipe":
                assert set(support) <= payload["sources"].keys()
                result = {"stages": [
                              {"script": "train.py", "name": "train", "outputs": ["prediction.txt", "results.txt"]},
                              {"script": "evaluate.py", "name": "score", "outputs": ["results.txt"]}],
                          "source_files": ["train.py", "evaluate.py", *support],
                          "seed_paths": [], "attestations": [], "readonly_directories": [],
                          "input_files": ["data/frozen.csv"], "metrics_file": "results.txt",
                          "metrics_format": "key_value", "metrics_append": True,
                          "metrics": [{"name": "accuracy", "keys": ["Accuracy - mean"]}], "questions": []}
                if not self.checked_question:
                    result["questions"] = ["Please send config/pilot.ini."]
            else:
                assert role == "onboard-native-review"
                needs_question = bool(payload["recipe"]["questions"])
                self.checked_question = self.checked_question or needs_question
                result = {"accepted": not needs_question,
                          "findings": ["config/pilot.ini is in sources"] if needs_question else []}
            return response.model_validate(result), {"role": role}

    worker = Worker()
    interpreter = Path(sys.executable)
    pending = qualify(project, run, "Run the fixture", interpreter=interpreter, worker=worker)
    assert pending["phase"] == "awaiting_input"
    assert worker.checked_question
    assert [(issue["field"], issue["detail"]) for issue in pending["issues"]] == [("input_files", "data/frozen.csv")]
    assert not (run / "native-stages").exists()
    require_bubblewrap()
    (artifacts / "data").mkdir(parents=True)
    (artifacts / "data/frozen.csv").write_text("3")
    complete = qualify(project, run, "Run the fixture", artifacts=artifacts, worker=worker)
    assert complete["phase"] == "baseline_verified", complete
    assert complete["metrics"] == {"accuracy": 6}
    assert not (project / "results.txt").exists()
    assert not (project / "data/frozen.csv").exists()
    recipe = NativeRecipe.model_validate(Evidence(run).get(complete["recipe"])["content"]["recipe"])
    receipts = [Evidence(run).get(ref)["content"] for ref in complete["retained"]]
    train, score = (Path(row["workspace"]) for row in receipts)
    assert (train / "results.txt").read_text().count("Accuracy - mean:") == 1
    assert (score / "results.txt").read_text().count("Accuracy - mean:") == 2
    sources = source_cards(project, list(files), set(files))
    reused, metrics = run_recipe(project, run, Path(complete["interpreter"]), artifacts, recipe,
                                 {card["path"]: card for card in sources})
    assert reused == receipts and metrics == {"accuracy": 6}


@pytest.mark.parametrize(("new", "error", "message"), [
    ("Accuracy - mean: 0.7\n", FrozenInputChanged, "rewrote"),
    ("", ValueError, "append new"),
    ("Metadata: score stage\n", ValueError, "did not return"),
    ("Accuracy - mean: 0.8\nAccuracy - mean: 0.9\n", ValueError, "Duplicate"),
    ("Accuracy - mean: nan\n", ValueError, "non-numeric"),
])
def test_append_only_result_rejects_rewritten_stale_or_ambiguous_metrics(tmp_path, new, error, message):
    prior = tmp_path / "train"
    final = tmp_path / "score"
    prior.mkdir()
    final.mkdir()
    prefix = "Accuracy - mean: 0.6\n"
    (prior / "results.txt").write_text(prefix)
    (final / "results.txt").write_text(new if error is FrozenInputChanged else prefix + new)
    recipe = NativeRecipe(stages=[], source_files=[], seed_paths=[], attestations=[], readonly_directories=[],
                          metrics_file="results.txt", metrics=[{"name": "accuracy", "keys": ["Accuracy - mean"]}],
                          questions=[], metrics_format="key_value", metrics_append=True)
    with pytest.raises(error, match=message):
        read_metrics(final, prior, recipe)


def test_native_label_type_diagnostic_omits_training_labels(tmp_path):
    source = tmp_path / "train.py"
    source.write_text("pass\n")
    (tmp_path / "stderr.log").write_text(
        f'Traceback (most recent call last):\n  File "{source}", line 17, in main\n'
        "    validation_label = 'sensitive-class'\n"
        "TypeError: Labels in y_true and y_pred should be of the same type. "
        "Got y_true=['sensitive-class'] and y_pred=[0].\n"
    )
    failure = failure_context([{"stage": "train", "error": "Stage exited with code 1", "logs": str(tmp_path)}],
                              tmp_path)
    assert failure["error_type"] == "TypeError" and failure["label_type_mismatch"]
    assert failure["source_locations"] == [{"path": "train.py", "line": 17}]
    assert "sensitive-class" not in json.dumps(failure)


def test_onboard_execute_reuses_the_standard_contract_without_crossing_projects(tmp_path, existing_experiment):
    source, root = existing_experiment, tmp_path / "standard"
    objective = "Measure the existing command's quality"
    spec = {"objective": objective, "knowledge": "Use the repository's known, fixed data splits and metric.",
            "metrics": [{"name": "quality", "direction": "max", "threshold": 0.8}], "max_experiments": 1,
            "reversible": True, "affordable": True, "confirmed_knowledge": True,
            "command_experiment": {"command": [sys.executable, "evaluate.py"], "split_manifest": "splits.json",
                                   "baseline": {"x": 0}, "choices": {"x": [0, 1]}}}
    (source / "experiment.json").write_text(json.dumps(spec))
    cli = CliRunner()
    suffix = [str(root), "--objective", objective, "--execute"]
    passed = cli.invoke(app, ["onboard", str(source), *suffix])
    assert passed.exit_code == 0, passed.output
    assert '"phase": "baseline_verified"' in passed.stdout
    assert cli.invoke(app, ["onboard", str(source), *suffix]).exit_code == 0
    other = tmp_path / "other-project"
    shutil.copytree(source, other)
    denied = cli.invoke(app, ["onboard", str(other), *suffix])
    assert denied.exit_code == 2, denied.output
