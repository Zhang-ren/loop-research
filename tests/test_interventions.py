import pytest

from loop_research.engine import intervention_of, intervention_targets
from loop_research.training import BASELINE_CODE


def test_two_classifier_parameters_are_not_one_variable():
    candidate = BASELINE_CODE.replace("C=1.0", "C=0.1").replace("max_iter=500", "max_iter=700")
    with pytest.raises(ValueError, match="Multiple estimator"):
        intervention_of(BASELINE_CODE, candidate)


def test_actual_parameter_identity_survives_hypothesis_label_changes():
    first = intervention_of(BASELINE_CODE, BASELINE_CODE.replace("C=1.0", "C=0.1"))
    second = intervention_of(BASELINE_CODE, BASELINE_CODE.replace("C=1.0", "C=10.0"))
    other = intervention_of(BASELINE_CODE, BASELINE_CODE.replace("random_state=seed", "random_state=seed, class_weight='balanced'"))
    assert first == second and first != other


def test_multiple_preprocessing_insertions_rejected():
    code = BASELINE_CODE.replace("from sklearn.pipeline", "from sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\nfrom sklearn.pipeline").replace("make_pipeline(LogisticRegression", "make_pipeline(StandardScaler(), PCA(n_components=10), LogisticRegression")
    with pytest.raises(ValueError, match="Multiple preprocessing"):
        intervention_of(BASELINE_CODE, code)


def test_replacing_a_preprocessor_family_requires_an_existing_preprocessor():
    assert 'preprocessing.family' not in intervention_targets(BASELINE_CODE)
    scaled = BASELINE_CODE.replace('from sklearn.pipeline', 'from sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline').replace('make_pipeline(LogisticRegression', 'make_pipeline(StandardScaler(), LogisticRegression')
    assert 'preprocessing.family' in intervention_targets(scaled)
