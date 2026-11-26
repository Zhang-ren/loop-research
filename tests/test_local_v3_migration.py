"""Regression checks for inputs observed in a real local migration."""
import io
import json
import tarfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from loop_research.command_experiment import read_splits, split_view
from loop_research.onboard_replay import ReplayRecipe, ReplayStep, script_path, seed_historical_inputs
from loop_research.workers import response_schema
from loop_research.onboarding import OnboardingPlan


class LocalMigrationTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.temp = Path(directory.name)

    def archive(self, value):
        path = self.temp / "repo.tar"
        payload = json.dumps(value).encode()
        with tarfile.open(path, "w") as archive:
            member = tarfile.TarInfo("splits.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        return path

    def test_loso_permits_subject_to_change_role_between_folds(self):
        source = {"folds": [{"id": "f1", "train": ["a"], "validation": ["b"]},
                            {"id": "f2", "train": ["b"], "validation": ["a"]}],
                  "heldout": None}
        splits = read_splits(self.archive(source), "splits.json")
        visible, train, evaluation = split_view(splits, "validation")
        self.assertEqual(visible, {"folds": source["folds"]})
        self.assertFalse(set(train) & set(evaluation))
        with self.assertRaisesRegex(ValueError, "no independent holdout"):
            split_view(splits, "test")

    def test_loso_rejects_a_used_subject_as_independent_test(self):
        source = {"folds": [{"id": "f1", "train": ["a"], "validation": ["b"]}],
                  "heldout": {"train": ["a"], "test": ["b"]}}
        with self.assertRaisesRegex(ValueError, "overlap the explored folds"):
            read_splits(self.archive(source), "splits.json")

    def test_recipe_paths_and_newly_generated_evidence(self):
        source, scratch = self.temp / "source", self.temp / "scratch"
        run_path = source / "experiment" / "output"
        run_path.mkdir(parents=True)
        (run_path / "old.json").write_text("{}")
        (run_path / "verification.json").write_text("{}")
        recipe = ReplayRecipe(working_directory="experiment", copy_files=[], read_only_directories=[],
                              seed_directories=["experiment/output"], steps=[],
                              evidence_files=["experiment/output/verification.json"], prerequisites=[])
        for name in ("run.py", "experiment/run.py"):
            self.assertEqual(script_path(recipe, ReplayStep(script=name, arguments=[])).as_posix(),
                             "experiment/run.py")
        copied = seed_historical_inputs(source, scratch, recipe)
        self.assertEqual(list(copied), ["experiment/output/old.json"])
        self.assertFalse((scratch / recipe.evidence_files[0]).exists())

    def test_onboarding_script_paths_are_wire_enums(self):
        props = response_schema(OnboardingPlan, source_paths=["repo/run.py"])["properties"]
        self.assertEqual(props["baseline_entrypoints"]["items"]["enum"], ["repo/run.py"])
