import pytest
from pathlib import Path

from loop_research.engine import Research
from loop_research.models import Project
from loop_research.replay import reproduce
from loop_research.training import BASELINE_CODE


@pytest.fixture
def project():
    return Project.model_validate_json((Path(__file__).parents[1] / "examples/breast-cancer.json").read_bytes())


@pytest.fixture
def engine(tmp_path, project):
    result = Research.initialize(tmp_path, project)
    class NoWorker:
        calls = []
        def run(self, *args, **kwargs):
            raise AssertionError("Audit checks must not invoke a model")
    result.worker = NoWorker()
    return result


def test_tracking_rejects_misattributed_code(engine):
    baseline = engine.evidence.get(engine.store.read().baseline_evidence)
    with pytest.raises(ValueError, match="does not match"):
        engine.track("invalid", baseline["content"], BASELINE_CODE.replace("C=1.0", "C=9.0"),
                     engine.store.read().baseline_evidence)


def test_report_rejects_metrics_not_matching_raw_evidence(engine):
    engine.store.change(lambda s: s.best_metrics.update(accuracy=1.0), event="TEST corrupt projection")
    with pytest.raises(ValueError, match="original measurements"):
        engine.report()


def test_reproduce_uses_original_code_samples_and_predictions(engine):
    result = reproduce(engine, engine.store.read().baseline_evidence)
    assert result["passed"] and result["checks"]["same_predictions"]
    assert not engine.worker.calls


def test_archived_runtime_rejects_modified_controller_files(engine):
    from loop_research.reproducibility import archived_runtime, prepare_archived_runtime

    reference, manifest = archived_runtime(engine.root)
    files = manifest["source_sha256"]
    cached = prepare_archived_runtime(engine.root, reference, files)
    assert prepare_archived_runtime(engine.root, reference, files) == cached
    (cached / "loop_research" / "cli.py").write_bytes(b"# modified")
    with pytest.raises(ValueError, match="Archived runtime file changed: cli.py"):
        prepare_archived_runtime(engine.root, reference, files)
    (engine.root / "runtime-source" / "cli.py").write_bytes(b"# modified source")
    with pytest.raises(ValueError, match="Archived runtime file changed: cli.py"):
        prepare_archived_runtime(engine.root, reference, files)


def test_baseline_already_meeting_targets_does_not_spawn(tmp_path, project):
    value = project.model_dump()
    value["metrics"] = [{"name": "accuracy", "threshold": 0.8}, {"name": "roc_auc", "threshold": 0.9}]
    engine = Research.initialize(tmp_path, Project.model_validate(value))
    class NoWorker:
        def run(self, *args, **kwargs):
            raise AssertionError("No model call needed when baseline already meets targets")
    engine.worker = NoWorker()
    state = engine.tick()
    assert state.phase == "finalize" and state.used == 0 and not state.active_tasks
