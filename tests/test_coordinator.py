import pytest
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
from sklearn import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder, FunctionTransformer


@pytest.mark.parametrize(
    ("input_label",),
    (
            (0,),
            (1,),
    )
)
def test_coordinator_assert_classification_pos_label(coordinator, data_classification_balanced, input_label):
    """Testing if assessing provided classification_pos_label returns the provided label if its in y values."""
    y = data_classification_balanced[1]
    coordinator.y = y
    pos_label = 1
    pos_label = coordinator._check_classification_pos_label(input_label)
    assert pos_label == input_label


@pytest.mark.parametrize(
    ("error_label",),
    (
            ("3",),
            ("Vegetables",),
            (4,),
    )
)
def test_coordinator_assert_classification_pos_label_error(coordinator, data_classification_balanced, error_label):
    """Testing if coordinator raises an error when classification_pos_label is explicitly provided but it
    doesn't exist in y."""
    y = data_classification_balanced[1]
    coordinator.y = y
    with pytest.raises(ValueError) as excinfo:
        c = coordinator._check_classification_pos_label(error_label)

    assert str(error_label) in str(excinfo.value)


@pytest.mark.parametrize(
    ("warning_label",),
    (
            ("Fruits",),
            ("Dairy",),
            ("Sweets",),
    )
)
def test_coordinator_assert_classification_pos_label_warning(coordinator, data_multiclass, root_path_to_package, warning_label):
    """Testing if warning is raised when classification_pos_label is explicitly provided for multiclass y."""
    y = data_multiclass[1]
    coordinator.y = y
    coordinator._force_classification_pos_label_multiclass_flag = False
    pos_label = 1
    with pytest.warns(Warning) as warninfo:
        pos_label = coordinator._check_classification_pos_label(warning_label)
    assert "classification_pos_label will be ignored" in warninfo[0].message.args[0]
    assert pos_label is None


@pytest.mark.parametrize(
    ("input_label",),
    (
            ("Fruits",),
            ("Dairy",),
            ("Sweets",),
    )
)
def test_coordinator_assert_classification_pos_label_forced(coordinator, data_multiclass, root_path_to_package, input_label):
    """Testing if classification_pos_label is set correctly for multiclass y when flag for forcing it is set to True."""
    y = data_multiclass[1]
    coordinator.y = y
    coordinator._force_classification_pos_label_multiclass_flag = True
    pos_label = 1
    pos_label = coordinator._check_classification_pos_label(input_label)
    assert pos_label == input_label


@pytest.mark.parametrize(
    ("input_df", "limit", "expected_flag"),
    (
            (pd.DataFrame({"a": [1, 2], "b": [2, 3]}), 3, True),
            (pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]}), 3, True),
            (pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4], "e": [5]}), 4, False),
            (pd.DataFrame({"a": [1, 2], "b": [1, 2]}), 1, False)
    )
)
def test_coordinator_assess_n_features(coordinator, input_df, limit, expected_flag):
    """Testing if assessing dataframes for the number of features correctly sets the flag (based on the limit)."""
    coordinator._n_features_pairplots_limit = limit
    coordinator._assess_n_features(input_df)

    assert coordinator._create_pairplots_flag == expected_flag


def test_coordinator_create_test_splits(coordinator, data_classification_balanced, seed):
    """Testing if the train/test split in coordinator is done correctly."""
    X, y = data_classification_balanced

    splitter = StratifiedShuffleSplit(random_state=seed)
    splitter.get_n_splits(X, y)

    train_indexes, test_indexes = next(splitter.split(X, y))
    X_train, X_test, y_train, y_test = X.loc[train_indexes], X.loc[test_indexes], y.loc[train_indexes], y.loc[test_indexes]
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    coordinator._create_test_splits()
    assert coordinator.X_train.equals(X_train)
    assert coordinator.X_test.equals(X_test)
    assert coordinator.y_train.equals(y_train)
    assert coordinator.y_test.equals(y_test)


def test_coordinator_fit_transform_test_splits(coordinator, data_classification_balanced, seed):
    """Testing if fit_transform_test_splits does fitting on train data and the transforms splits appropriately."""
    c = coordinator
    c._create_test_splits()
    transformer_X = clone(c.transformer_eval.preprocessor_X)
    transformer_y = clone(c.transformer_eval.preprocessor_y)

    transformer_X.fit(c.X_train)
    transformer_y.fit(c.y_test)

    expected_X_train = transformer_X.transform(c.X_train)
    expected_X_test = transformer_X.transform(c.X_test)
    expected_y_train = transformer_y.transform(c.y_train)
    expected_y_test = transformer_y.transform(c.y_test)

    c._fit_transform_test_splits()

    assert np.array_equal(c.transformed_X_train.toarray(), expected_X_train.toarray())
    assert np.array_equal(c.transformed_X_test.toarray(), expected_X_test.toarray())
    assert np.array_equal(c.transformed_y_train, expected_y_train)
    assert np.array_equal(c.transformed_y_test, expected_y_test)


def test_coordinator_set_custom_transformer(coordinator, data_classification_balanced):
    """Testing if setting custom transformers update both instances of regular and train/test splits Transformers."""
    numerical_tr = [SimpleImputer(strategy="mean"), PowerTransformer()]
    categorical_tr = [SimpleImputer(strategy="constant", fill_value="Missing"), OneHotEncoder(drop="first")]
    y_transformer = FunctionTransformer()

    coordinator.set_custom_transformers(categorical_tr, numerical_tr, y_transformer)

    for tr in [coordinator.transformer, coordinator.transformer_eval]:
        assert tr.numerical_transformers == numerical_tr
        assert tr.categorical_transformers == categorical_tr
        assert tr.y_transformer == y_transformer


# TODO: test if indexes match between untransformed and transformed data


