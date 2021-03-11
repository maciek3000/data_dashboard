from create_model.features import NumericalFeature, CategoricalFeature, Features, sort_strings
import pandas as pd
import pytest


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
def test_categorical_feature_create_raw_mapping(
        data_classification_balanced, expected_raw_mapping, column_name
):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)

    series = df[column_name]
    feature = CategoricalFeature(series, column_name, "test", False)

    actual_mapping = feature._create_raw_mapping()
    expected_mapping = expected_raw_mapping[column_name]

    assert actual_mapping == expected_mapping


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
def test_categorical_feature_create_mapped_series(
        data_classification_balanced, expected_raw_mapping, column_name
):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)

    expected_series = df[column_name].copy().replace(expected_raw_mapping[column_name])

    series = df[column_name]
    feature = CategoricalFeature(series, column_name, "test", False)
    actual_series = feature._create_mapped_series()

    assert actual_series.equals(expected_series)


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
def test_categorical_features_create_descriptive_mapping(
        data_classification_balanced, expected_mapping, column_name, feature_descriptor
):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)

    series = df[column_name]
    feature = CategoricalFeature(series, column_name, "test", False, mapping=feature_descriptor.mapping(column_name))
    actual_mapping = feature._create_descriptive_mapping()

    assert actual_mapping == expected_mapping[column_name]


def test_features_raw_mapping(fixture_features, expected_raw_mapping, categorical_features):
    """Testing if the .raw_mapping() of CategoricalFeature works correctly."""
    f = fixture_features
    actual_raw_mapping = {feature: f[feature].raw_mapping for feature in categorical_features}

    assert actual_raw_mapping == expected_raw_mapping


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

