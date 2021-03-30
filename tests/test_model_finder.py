import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, PassiveAggressiveClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.dummy import DummyClassifier, DummyRegressor

from ml_dashboard.model_finder import name, ModelFinder, ModelNotSetError
from ml_dashboard.models import classifiers, regressors


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
    assert np.array_equal(actual_model.predict(test_input), np.array([median] * 6))


def test_model_finder_classification_search():
    assert False


@pytest.mark.parametrize(
    ("mode",),
    (
            ("QUICK",),
            (10,),
            (None,),
            (False,),
            (lambda x: "quick",),
    )
)
def test_model_finder_search_incorrect_mode(model_finder_classification, mode):
    categories = ", ".join(model_finder_classification._modes)
    with pytest.raises(ValueError) as excinfo:
        model_finder_classification.search(mode=mode)
    assert categories in str(excinfo.value)


@pytest.mark.parametrize(
    ("incorrect_model",),
    (
            (Ridge(),),
            ("LinearRegression",),
            (10,),
            (lambda x: x,),
    )
)
def test_model_finder_search_incorrect_model(model_finder_classification, incorrect_model):
    mode = model_finder_classification._mode_quick
    with pytest.raises(ValueError) as excinfo:
        model_finder_classification.search(models=incorrect_model, mode=mode)
    assert "models should be Dict, List-like or None" in str(excinfo.value)


def test_model_finder_fit(model_finder_classification):
    assert False


def test_model_finder_fit_no_model(model_finder_classification):
    with pytest.raises(ModelNotSetError):
        model_finder_classification.fit()


def test_model_finder_predict(model_finder_classification):
    assert False


def test_model_finder_predict_no_model(model_finder_classification):
    with pytest.raises(ModelNotSetError):
        model_finder_classification.predict(["test_input"])


def test_model_finder_perform_gridsearch_classification(model_finder_classification, chosen_classifiers_grid, seed):
    expected_models = [
        (DecisionTreeClassifier, {"max_depth": 10, "criterion": "entropy", "random_state": seed}),
        (LogisticRegression, {"C": 1.0, "tol": 0.1, "random_state": seed}),
        (SVC, {"C": 0.1, "tol": 0.1, "random_state": seed})
    ]
    standard_keys = [
        "iter", "n_resources", "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params",
        "split0_train_score", "split1_train_score", "split2_train_score", "split3_train_score", "split4_train_score",
        "split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score", "split4_test_score",
        "rank_test_score", "mean_test_score", "mean_train_score", "std_test_score", "std_train_score"
    ]

    actual_models, actual_results = model_finder_classification._perform_gridsearch(
        chosen_classifiers_grid, roc_auc_score, cv=5
    )

    assert sorted(actual_models, key=lambda x: x[0].__name__) == expected_models
    assert len(actual_results.keys()) == len(expected_models)
    for model_tuple in expected_models:
        ml = model_tuple[0]
        params = model_tuple[1]
        ml_specific_keys = ["param_" + param for param in params.keys()] + standard_keys
        expected_keys = set(ml_specific_keys)
        actual_keys = set(actual_results[ml].keys())
        assert actual_keys == expected_keys


def test_model_finder_perform_gridsearch_regression(model_finder_regression, chosen_regressors_grid, seed):
    expected_models = [
        (DecisionTreeRegressor, {"max_depth": 10, "criterion": "mae", "random_state": seed}),
        (Ridge, {"alpha": 0.0001, "random_state": seed}),
        (SVR, {"C": 0.1, "tol": 1.0})
    ]

    standard_keys = [
        "iter", "n_resources", "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params",
        "split0_train_score", "split1_train_score", "split2_train_score", "split3_train_score", "split4_train_score",
        "split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score", "split4_test_score",
        "rank_test_score", "mean_test_score", "mean_train_score", "std_test_score", "std_train_score"
    ]

    actual_models, actual_results = model_finder_regression._perform_gridsearch(
        chosen_regressors_grid, mean_squared_error, cv=5
    )

    assert sorted(actual_models, key=lambda x: x[0].__name__) == expected_models
    assert len(actual_results.keys()) == len(expected_models)
    for model_tuple in expected_models:
        ml = model_tuple[0]
        params = model_tuple[1]
        ml_specific_keys = ["param_" + param for param in params.keys()] + standard_keys
        expected_keys = set(ml_specific_keys)
        actual_keys = set(actual_results[ml].keys())
        assert actual_keys == expected_keys


@pytest.mark.parametrize(
    ("input", "expected_result"),
    (
            ({
                 LogisticRegression: {"params_C": [10, 15], "score": [0.25, 0.30], "iter": [1, 2]},
                 DecisionTreeClassifier: {"iter": [1], "score": [0.45], "params_max_depth": [10]}
             },
             pd.DataFrame(data={
                     "params_C": [10, 15, np.nan],
                     "score": [0.25, 0.30, 0.45],
                     "iter": [1, 2, 1],
                     "params_max_depth": [np.nan, np.nan, 10],
                     "model": ["LogisticRegression", "LogisticRegression", "DecisionTreeClassifier"]
                 })

            ),
            ({
                Exception: {"name": ["exception"], "reason": ["none"], "test": ["done"]},
                Ridge: {"score": [3, 4, 5], "fit-time": [1, 1, 4], "test": [None, None, None]},
                pd.DataFrame: {"name": ["df1", "df2"], "reason": ["eggs", "eggs2"], "score": [4, -1], "fit-time": [0, np.nan]}
            },
            pd.DataFrame(data={
                "model": ["Exception", "Ridge", "Ridge", "Ridge", "DataFrame", "DataFrame"],
                "name": ["exception", np.nan, np.nan, np.nan, "df1", "df2"],
                "reason": ["none", np.nan, np.nan, np.nan, "eggs", "eggs2"],
                "score": [np.nan, 3, 4, 5, 4,  -1],
                "fit-time": [np.nan, 1, 1, 4, 0, np.nan],
                "test": ["done", None, None, None, np.nan, np.nan]
            })
            )

    )
)
def test_model_finder_update_gridsearch_results(model_finder_classification, input, expected_result):
    expected_result = expected_result.rename({"model": model_finder_classification._model_name})
    model_finder_classification._update_gridsearch_results(input)

    actual_result = model_finder_classification._gridsearch_results

    assert actual_result.equals(expected_result[actual_result.columns])


def test_model_finder_perform_quicksearch_classification(model_finder_classification, chosen_classifiers_grid, seed):
    expected_models = [
        (DecisionTreeClassifier, 0.5476190476190477),
        (LogisticRegression, 0.619047619047619),
        (SVC, 0.48809523809523814),
    ]
    expected_keys = {"fit_time", "roc_auc_score", "params"}

    actual_models, actual_results = model_finder_classification._perform_quicksearch(
        chosen_classifiers_grid, roc_auc_score
    )

    assert sorted(actual_models, key=lambda x: x[0].__name__) == expected_models
    assert len(actual_results.keys()) == len(expected_models)

    for model_tuple in expected_models:
        model = model_tuple[0]
        actual_keys = set(actual_results[name(model)].keys())
        assert actual_keys == expected_keys