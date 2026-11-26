from types import SimpleNamespace

import pytest

from loop_research.bridge_runtime import validate_predictions
from loop_research.onboard_adapter import compare_baseline
from loop_research.storage import canonical, digest


def test_equal_totals_do_not_admit_changed_event_predictions(tmp_path):
    path = tmp_path / "fold0-physical.json"
    expected_rows = [{"start_frame": 1, "end_frame": 10, "matched": True}]
    blob = canonical(expected_rows)
    path.write_bytes(blob)
    proposal = SimpleNamespace(historical_predictions_template="{fold}-{arm}.json")
    expected = {"tp": 1, "fn": 0, "f1": 1.0}
    splits = {"folds": [{"id": "fold0"}]}
    observed = {"metrics": expected, "fold_results": [{"id": "fold0", "event_predictions": expected_rows}]}
    seal = {path.name: digest(blob)}
    compare_baseline(tmp_path, proposal, "physical", expected, observed, splits, seal)
    observed["fold_results"][0]["event_predictions"] = [{"start_frame": 2, "end_frame": 10, "matched": True}]
    with pytest.raises(ValueError, match="per-event predictions differ"):
        compare_baseline(tmp_path, proposal, "physical", expected, observed, splits, seal)


def test_prediction_rejects_video_outside_fold():
    features = {"folds": [{"id": "fold0", "videos": [{"video_name": "held", "nframes": 20}]}]}
    proposal = {"video_name": "held", "start_frame": 1, "end_frame": 5, "score": 0.2, "type_idx": 2}
    predictions = {"folds": [{"id": "fold0", "proposals": [proposal]}]}
    validate_predictions(predictions, features)
    proposal["video_name"] = "another-fold"
    with pytest.raises(ValueError, match="outside the admitted input"):
        validate_predictions(predictions, features)
