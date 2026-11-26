import pytest

from loop_research.model_policy import validate_proposal
from loop_research.platform import LocalPlatform
from loop_research.training import BASELINE_CODE


@pytest.mark.parametrize('code', [
    BASELINE_CODE.replace('random_state=seed))', "random_state=seed), memory='/tmp/unauthorized-model-cache')"),
    BASELINE_CODE.replace('random_state=seed', 'random_state=seed, n_jobs=1000'),
    'from sklearn.ensemble import RandomForestClassifier\ndef build_model(seed):\n return RandomForestClassifier(n_estimators=513, random_state=seed)\n',
    'from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return make_pipeline(PolynomialFeatures(degree=100), LogisticRegression())\n',
    'from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return make_pipeline(PolynomialFeatures(3), PolynomialFeatures(3), LogisticRegression())\n',
])
def test_unsafe_or_unaffordable_factories_never_launch_training(tmp_path, code):
    platform=LocalPlatform(tmp_path)
    with pytest.raises(ValueError):
        platform.submit('TEST unsafe proposal',code,42)
    assert not list(platform.root.glob('*/request.json'))


def test_bounded_composition_can_reduce_dimension_before_another_expansion():
    code='from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.decomposition import PCA\nfrom sklearn.linear_model import LogisticRegression\ndef build_model(seed):\n return make_pipeline(PolynomialFeatures(2), PCA(n_components=3), PolynomialFeatures(2), LogisticRegression())\n'
    validate_proposal(code,42)
    validate_proposal(BASELINE_CODE,42)
