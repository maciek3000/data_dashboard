from create_model.features import NumericalFeature, CategoricalFeature, Features, sort_strings
import pandas as pd
import pytest

def test_features_analyze_features(data_classification_balanced, feature_descriptor):
    """Testing if .analyze_features() method of Features class returns a dictionary with a correct content"""
    n = NumericalFeature
    c = CategoricalFeature
    expected = {
        "Sex": c,
        "AgeGroup": c,
        "Height": n,
        "Product": c,
        "Price": n,
        "bool": c,
        "Target": c
    }

    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    f.original_dataframe = pd.concat([X, y], axis=1)  # original_dataframe needs to be set up
    actual = f._analyze_features(feature_descriptor)

    assert isinstance(actual, dict)
    for key, item in expected.items():
        assert isinstance(actual[key], item)


def test_features_raw_mapping(fixture_features, expected_raw_mapping, categorical_features):
    """Testing if the .raw_mapping() of CategoricalFeature works correctly."""
    f = fixture_features
    actual_raw_mapping = {feature: f[feature].raw_mapping for feature in categorical_features}

    assert actual_raw_mapping == expected_raw_mapping

@pytest.mark.parametrize(
    ("column_name",),
    (
            ("AgeGroup",),
            ("bool",),
            ("Product",),
            ("Sex",),
            ("Target",)
    )
)
def test_categorical_feature_raw_mapping(
        data_classification_balanced, categorical_features, expected_raw_mapping, column_name
):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]

    df = pd.concat([X, y], axis=1)
    series = df[column_name]
    feature = CategoricalFeature(series, column_name, "test", False)

    actual_mapping = feature._create_raw_mapping()
    expected_mapping = expected_raw_mapping[column_name]

    assert actual_mapping == expected_mapping
