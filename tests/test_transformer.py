from create_model.transformer import Transformer
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder, QuantileTransformer, StandardScaler
import pytest
import numpy as np
import pandas as pd


def test_transformer_create_preprocessor_X(categorical_features, numerical_features):
    """Testing if X preprocessor correctly assigns steps to columns depending on their type."""
    categorical_features.remove("Target")
    tr = Transformer(categorical_features, numerical_features, "Categorical")
    preprocessor = tr._create_preprocessor_X()

    expected_steps = [("numerical", numerical_features), ("categorical", categorical_features)]

    actual_steps = [(item[0], item[2]) for item in preprocessor.transformers]

    for step in expected_steps:
        assert step in actual_steps

    assert len(actual_steps) == len(expected_steps)


@pytest.mark.parametrize(
    ("target_type", "expected_function"),
    (
            ("Categorical", LabelEncoder()),
            ("Numerical", FunctionTransformer(lambda x: x))
    )
)
def test_transformer_create_preprocessor_y(categorical_features, numerical_features, target_type, expected_function):
    """Testing if y preprocessor is created correctly."""
    tr = Transformer(categorical_features, numerical_features, target_type)
    preprocessor = tr._create_preprocessor_y()

    assert type(preprocessor).__name__ == type(expected_function).__name__


@pytest.mark.parametrize(
    ("target_type",),
    (
            (None,),
            ("Target",),
            ("categorical",),
            (10,),
            (True,),
            (np.nan,),
    )
)
def test_transformer_create_preprocessor_y_invalid_target_type(categorical_features, numerical_features, target_type):
    """Testing if ._create_preprocessor_y raises an Exception when invalid target_type is provided"""
    tr = Transformer(categorical_features, numerical_features, "Categorical")  # initiating with proper type
    tr.target_type = target_type
    with pytest.raises(ValueError) as excinfo:
        preprocessor = tr._create_preprocessor_y()
    assert "should be Categorical or Numerical" in str(excinfo.value)


@pytest.mark.parametrize(
    ("feature_name",),
    (
            ("AgeGroup",),
            ("bool",),
            ("Product",),
            ("Sex",),
            ("Target",),
    )
)
def test_transformer_transform_y_categorical(
        data_classification_balanced, categorical_features, numerical_features, expected_raw_mapping, feature_name
):
    """Testing if fit_y() and transform_y() are changing provided y correctly (when y is categorical)"""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)

    target = df[feature_name]
    mapping = {key: int(item - 1) for key, item in expected_raw_mapping[feature_name].items()}
    mapping[np.nan] = max(mapping.values()) + 1
    expected_result = target.replace(mapping).array

    tr = Transformer(categorical_features, numerical_features, "Categorical")
    actual_result = tr.fit_transform_y(target)

    assert np.array_equal(actual_result, expected_result)


@pytest.mark.parametrize(
    ("feature_name",),
    (
            ("Height",),
            ("Price",),
    )
)
def test_transformer_transform_y_numerical(
        data_classification_balanced, categorical_features, numerical_features, feature_name
):
    """Testing if fit_y() and transform_y() are changing provided y correctly (when y is numerical)"""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    target = df[feature_name]
    expected_result = target.array

    tr = Transformer(categorical_features, numerical_features, "Numerical")
    actual_result = tr.fit_transform_y(target)

    assert np.allclose(actual_result, expected_result, equal_nan=True)


@pytest.mark.parametrize(
    ("feature_name", "csr_matrix_flag"),
    (
            ("AgeGroup", True),
            ("bool", False),
            ("Product", True),
            ("Sex", False),
            ("Target", False),
    )
)
def test_transformer_transform_X_categorical(data_classification_balanced, feature_name, csr_matrix_flag):
    """Testing if every categorical column from a test data is transformed correctly."""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    # replacing for SimpleImputer which cant handle bool dtype
    df["bool"] = df["bool"].replace({False: 0, True: 1})
    feature = df[feature_name]
    most_frequent = feature.value_counts(dropna=False).index[0]
    feature = feature.fillna(most_frequent)
    expected_result = OneHotEncoder(handle_unknown="ignore").fit_transform(feature.to_numpy().reshape(-1, 1)).toarray()

    tr = Transformer([feature_name], [], "Categorical")
    actual_result = tr.fit_transform(pd.DataFrame(df[feature_name]))

    # for n > 2 unique values, output is a csr_matrix
    if csr_matrix_flag:
        actual_result = actual_result.toarray()

    assert pd.DataFrame(actual_result).equals(pd.DataFrame(expected_result))


@pytest.mark.parametrize(
    ("feature_name",),
    (
            ("Height",),
            ("Price",),
    )
)
def test_transformer_transform_X_numerical(data_classification_balanced, feature_name):
    """Testing if every numerical column from a test data is transformed correctly."""
    random_state = 1
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    feature = df[feature_name]
    median = feature.describe()["50%"]
    feature = feature.fillna(median)
    feature = QuantileTransformer(output_distribution="normal", random_state=random_state ).fit_transform(feature.to_numpy().reshape(-1, 1))
    expected_result = StandardScaler().fit_transform(feature)

    tr = tr = Transformer([], [feature_name], "Categorical", random_state=random_state)
    actual_result = tr.fit_transform(pd.DataFrame(df[feature_name]))

    assert np.allclose(actual_result, expected_result)