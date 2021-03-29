import pytest
from create_model.model_finder import name, ModelFinder
from sklearn.linear_model import Ridge, PassiveAggressiveClassifier


@pytest.mark.parametrize(
    ("obj", "expected_result"),
    (
            (dict(), "dict"),
            (dict, "dict"),
            (None, "NoneType"),
            (lambda x: x, "<lambda>"),
            (Ridge, "Ridge"),
            (Ridge(), "Ridge"),
            (PassiveAggressiveClassifier, "PassiveAggressiveClassifier"),
            (PassiveAggressiveClassifier(), "PassiveAggressiveClassifier")

    )
)
def test_name(obj, expected_result):
    actual_result = name(obj)
    assert actual_result == expected_result


def test_model_finder_init(category):
    assert True
