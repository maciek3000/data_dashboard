import pytest
import numpy as np
from sklearn.linear_model import Ridge, PassiveAggressiveClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.dummy import DummyClassifier, DummyRegressor

from create_model.model_finder import name, ModelFinder
from create_model.models import classifiers, regressors


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


@pytest.mark.parametrize(
    ("category_type",),
    (
            ("categorical",),
            ("numerical",),
    )
)
def test_model_finder_init(data_classification_balanced, split_dataset_categorical, seed, category_type):
    if category_type == "categorical":
        expected_problem = ModelFinder._classification
        expected_scorings = ModelFinder._scoring_classification
        expected_models = classifiers
        expected_default_scoring = roc_auc_score
    elif category_type == "numerical":
        expected_problem = ModelFinder._regression
        expected_scorings = ModelFinder._scoring_regression
        expected_models = regressors
        expected_default_scoring = mean_squared_error
    else:
        raise

    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    mf = ModelFinder(
        X,
        y,
        *split_dataset_categorical,
        target_type=category_type,
        random_state=seed
    )

    assert mf.problem == expected_problem
    assert mf.scoring_functions == expected_scorings
    assert mf.default_models == expected_models
    assert mf.default_scoring == expected_default_scoring


@pytest.mark.parametrize(
    ("category_type",),
    (
            ("Categorical",),
            (None,),
            (10,),
            (lambda x: x,),
            ("regression",),
    )
)
def test_model_finder_init_improper_problem_type(
        data_classification_balanced, split_dataset_categorical, seed, category_type
):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    err_msg = "Expected one of the categories: "
    with pytest.raises(ValueError) as excinfo:
        mf = ModelFinder(
            X,
            y,
            *split_dataset_categorical,
            target_type=category_type,
            random_state=seed
        )
    assert err_msg in str(excinfo.value)


@pytest.mark.parametrize(
    ("test_input",),
    (
            ([0, 1, 2, 3, 4, 5],),
            ([6, 7, 8, 9, 10, 11],),
            ([12, 13, 14, 15, 16, 17],),
    )
)
def test_model_finder_dummy_categorical(model_finder_classification, data_classification_balanced, seed, test_input):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    expected_model = DummyClassifier(strategy="stratified", random_state=seed)
    expected_model.fit(X, y)



    actual_model, actual_model_scores = model_finder_classification._create_dummy_model()

    assert str(actual_model) == str(expected_model)
    assert np.array_equal(actual_model.predict(test_input), expected_model.predict(test_input))


@pytest.mark.parametrize(
    ("test_input",),
    (
            ([0, 1, 2, 3, 4, 5],),
            ([6, 7, 8, 9, 10, 11],),
            ([12, 13, 14, 15, 16, 17],),
    )
)
def test_model_finder_dummy_regression(model_finder_regression, test_input):
    expected_model = DummyRegressor(strategy="median")

    median = 35.619966243279364

    actual_model, actual_model_scores = model_finder_regression._create_dummy_model()

    assert str(actual_model) == str(expected_model)
    assert np.array_equal(actual_model.predict(test_input), np.array([median]*6))