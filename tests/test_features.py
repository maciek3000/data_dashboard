from ml_dashboard.features import NumericalFeature, CategoricalFeature, Features
from ml_dashboard.descriptor import FeatureDescriptor
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
    """Testing if ._create_raw_mapping function correctly extracts unique values in the Series and maps them."""
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
    """Testing if Series values are correctly replaced with a "raw" mapping."""
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
    """Testing if ._create_descriptive_mapping() correctly creates mapping between raw mapping and descriptions."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)

    series = df[column_name]
    feature = CategoricalFeature(series, column_name, "test", False, mapping=feature_descriptor.mapping(column_name))
    actual_mapping = feature._create_descriptive_mapping()

    assert actual_mapping == expected_mapping[column_name]


@pytest.mark.parametrize(
    ("column_name",),
    (
            ("AgeGroup",),
            ("Target",)
    )
)
def test_categorical_features_create_descriptive_mapping_changed_keys(
        data_classification_balanced, feature_descriptor_broken, expected_mapping, column_name
):
    """Testing if ._create_descriptive_mapping() creates correct output when the keys are incorrect:
        descriptions provided as str, yet the data itself is int/float."""

    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)

    series = df[column_name]
    feature = CategoricalFeature(series, column_name, "test", False,
                                 mapping=feature_descriptor_broken.mapping(column_name))
    actual_mapping = feature._create_descriptive_mapping()

    assert actual_mapping == expected_mapping[column_name]


@pytest.mark.parametrize(
    ("column_name",),
    (
            ("Height",),
            ("Price",),
    )
)
def test_numerical_features_no_mapping(
        data_classification_balanced, column_name
):
    """Testing if .mapping() from NumericalFeature returns None."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)

    series = df[column_name]
    feature = NumericalFeature(series, column_name, "test", False)

    assert feature.mapping() is None


@pytest.mark.parametrize(
    ("column_name", "expected_type"),
    (
            ("AgeGroup", "categorical"),
            ("bool", "categorical"),
            ("Date", "date"),
            ("Height", "numerical"),
            ("Price", "numerical"),
            ("Product", "categorical"),
            ("Sex", "categorical"),
            ("Target", "categorical")
    )
)
def test_features_impute_column_type(data_classification_balanced, column_name, expected_type):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)
    f = Features(X, y)

    cat = f.categorical
    num = f.numerical
    dat = f.date

    if expected_type == "categorical":
        expected = cat
    elif expected_type == "numerical":
        expected = num
    elif expected_type == "date":
        expected = dat
    else:
        raise

    actual = f._impute_column_type(df[column_name])

    assert actual == expected


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


def test_features_analyze_features_forced_category(data_classification_balanced, feature_descriptor_forced_categories):
    """Testing if .analyze_features() method of Features class returns a dictionary with a correct content
        when categories are forced by the FeatureDescriptor"""
    n = NumericalFeature
    c = CategoricalFeature
    expected = {
        "Sex": c,
        "AgeGroup": n,
        "Height": c,
        "Product": c,
        "Price": c,
        "bool": n,
        "Target": n
    }

    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor_forced_categories)

    f.original_dataframe = pd.concat([X, y], axis=1)  # original_dataframe needs to be set up
    actual = f._analyze_features(feature_descriptor_forced_categories)

    assert isinstance(actual, dict)
    for key, item in expected.items():
        assert isinstance(actual[key], item)


@pytest.mark.parametrize(
    ("feature_descriptor_type",),
    (
            ("normal",),
            ("forced",)
    )
)
def test_features_create_features(
        data_classification_balanced, feature_descriptor_type, feature_descriptor, feature_descriptor_forced_categories
):
    """Testing if ._create_features() returns correct values depending on the Features provided."""
    expected = ["AgeGroup", "bool", "Height", "Price", "Product", "Sex", "Target"]
    X, y = data_classification_balanced

    # couldn't find a way to incorporate fixtures into @pytest.mark.parametrize
    if feature_descriptor_type == "normal":
        fd = feature_descriptor
    elif feature_descriptor_type == "forced":
        fd = feature_descriptor_forced_categories
    else:
        raise

    f = Features(X, y, fd)

    actual = f._create_features()

    assert actual == expected


@pytest.mark.parametrize(
    ("feature_descriptor_type",),
    (
            ("normal",),
            ("forced",)
    )
)
def test_features_features_list_no_target(
        data_classification_balanced, feature_descriptor_type, feature_descriptor, feature_descriptor_forced_categories
):
    """Testing if .features() returns correct values when drop_target = True (without Target feature name)."""
    expected = ["AgeGroup", "bool", "Height", "Price", "Product", "Sex"]
    X, y = data_classification_balanced

    # couldn't find a way to incorporate fixtures into @pytest.mark.parametrize
    if feature_descriptor_type == "normal":
        fd = feature_descriptor
    elif feature_descriptor_type == "forced":
        fd = feature_descriptor_forced_categories
    else:
        raise

    f = Features(X, y, fd)
    actual = f.features(drop_target=True)

    assert actual == expected


@pytest.mark.parametrize(
    ("feature_descriptor_type", "expected"),
    (
            ("normal", ["Height", "Price"]),
            ("forced", ["AgeGroup", "bool",  "Target"])
    )
)
def test_features_create_numerical_features(
        data_classification_balanced, feature_descriptor_type, expected,
        feature_descriptor, feature_descriptor_forced_categories
):
    """Testing if ._create_numerical_features() returns correct values depending on the Features provided."""
    X, y = data_classification_balanced

    # couldn't find a way to incorporate fixtures into @pytest.mark.parametrize
    if feature_descriptor_type == "normal":
        fd = feature_descriptor
    elif feature_descriptor_type == "forced":
        fd = feature_descriptor_forced_categories
    else:
        raise

    f = Features(X, y, fd)

    actual = f._create_numerical_features()

    assert actual == expected


@pytest.mark.parametrize(
    ("feature_list", "target", "expected"),
    (
            (["Height", "Price"], "Target", ["Height", "Price"]),
            (["Height", "Price", "Target"], "Target", ["Height", "Price"]),
            (["Age", "Sex", "Target", "Zzz"], "Sex", ["Age", "Target", "Zzz"])
    )
)
def test_features_numerical_features_no_target(
        feature_list, target, expected, data_classification_balanced, feature_descriptor
):
    """Testing if .numerical_features() returns correct values when drop_target = True (without Target feature name)."""
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    f._numerical_features = feature_list
    f.target = target
    actual = f.numerical_features(drop_target=True)

    assert actual == expected


@pytest.mark.parametrize(
    ("feature_descriptor_type", "expected"),
    (
            ("normal", ["AgeGroup", "bool", "Product", "Sex", "Target"]),
            ("forced", ["Height", "Price", "Product", "Sex"])
    )
)
def test_features_create_categorical_features(
        data_classification_balanced, feature_descriptor_type, expected,
        feature_descriptor, feature_descriptor_forced_categories
):
    """Testing if ._create_categorical_features() returns correct values depending on the Features provided."""
    X, y = data_classification_balanced

    # couldn't find a way to incorporate fixtures into @pytest.mark.parametrize
    if feature_descriptor_type == "normal":
        fd = feature_descriptor
    elif feature_descriptor_type == "forced":
        fd = feature_descriptor_forced_categories
    else:
        raise

    f = Features(X, y, fd)

    actual = f._create_categorical_features()

    assert actual == expected


@pytest.mark.parametrize(
    ("feature_list", "target", "expected"),
    (
            (["Age", "Sex"], "Target", ["Age", "Sex"]),
            (["Age", "Sex", "Target"], "Target", ["Age", "Sex"]),
            (["Height", "Price", "Target", "Zzz"], "Height", ["Price", "Target", "Zzz"])
    )
)
def test_features_categorical_features_no_target(
        feature_list, target, expected, data_classification_balanced, feature_descriptor
):
    """Testing if .categorical_features() returns correct values when drop_target = True (without Target feature
    name). """
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    f._categorical_features = feature_list
    f.target = target
    actual = f.categorical_features(drop_target=True)

    assert actual == expected


def test_features_unused_features(data_classification_balanced, feature_descriptor):
    """Testing if unused_features() returns correct values."""
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    assert f.unused_features() == ["Date"]


def test_features_create_raw_dataframe(data_classification_balanced, feature_descriptor):
    """Testing if .create_raw_dataframe returns correct dataframe (the same that was provided as input to the
    object). """
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    expected_df = pd.concat([X, y], axis=1).drop(["Date"], axis=1)
    cols = expected_df.columns

    actual_df = f._create_raw_dataframe()[cols]

    assert actual_df.equals(expected_df)


def test_features_create_raw_dataframe_preserving_index(data_classification_balanced, feature_descriptor):
    """Testing if create_raw_dataframe preserves the index of the DataFrame."""
    X, y = data_classification_balanced
    not_expected_df = pd.concat([X, y], axis=1).drop(["Date"], axis=1)

    length = X.shape[0]
    new_ind = list(range(100, length + 100))
    X.index = new_ind
    y.index = new_ind
    f = Features(X, y, feature_descriptor)
    expected_df = pd.concat([X, y], axis=1).drop(["Date"], axis=1)
    expected_df.index = new_ind

    cols = expected_df.columns
    actual_df = f._create_raw_dataframe()[cols]

    assert not actual_df.equals(not_expected_df)
    assert actual_df.equals(expected_df)


def test_features_raw_data_no_target(data_classification_balanced, feature_descriptor):
    """Testing if raw_dataframe() drops Target column when drop_target=True."""
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    expected_df = X.drop(["Date"], axis=1)
    cols = expected_df.columns

    actual_df = f.raw_data(drop_target=True)[cols]

    assert actual_df.equals(expected_df)


def test_features_create_mapped_dataframe(data_classification_balanced, feature_descriptor, expected_raw_mapping):
    """Testing if ._create_mapped_dataframe correctly returns mapped dataframe (with replaced values according to
    mapping). """
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    expected_df = pd.concat([X, y], axis=1).drop(["Date"], axis=1).replace(expected_raw_mapping)
    cols = expected_df.columns

    actual_df = f._create_mapped_dataframe()[cols]

    assert actual_df.equals(expected_df)


def test_features_data(data_classification_balanced, feature_descriptor, expected_raw_mapping):
    """Testing if .data() returns mapped df (with replaced values according to mapping) but without Target column (
    when drop_target=True). """
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    expected_df = X.drop(["Date"], axis=1).replace(expected_raw_mapping)
    cols = expected_df.columns

    actual_df = f.data(drop_target=True)[cols]

    assert actual_df.equals(expected_df)


def test_features_create_mapping(data_classification_balanced, feature_descriptor, expected_mapping):
    """Testing if ._create_mapping() creates a correct mapping dictionary."""
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    expected = expected_mapping
    for feat in ["Height", "Price"]:
        expected[feat] = None

    actual = f.mapping()

    assert actual == expected


def test_features_create_descriptions(data_classification_balanced, feature_descriptions, feature_descriptor):
    """Testing if ._create_descriptions creates a correct descriptions dictionary."""
    placeholder = Features._description_not_available
    d = FeatureDescriptor._description
    assert True
    expected_descriptions = {}
    for feat in ["Sex", "Height", "Product", "Price", "bool", "Target"]:
        expected_descriptions[feat] = feature_descriptions[feat][d]

    expected_descriptions["AgeGroup"] = placeholder

    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)

    actual_descriptions = f.descriptions()

    assert actual_descriptions == expected_descriptions
