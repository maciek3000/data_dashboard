import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, PassiveAggressiveClassifier, LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.dummy import DummyClassifier, DummyRegressor

from ml_dashboard.model_finder import obj_name, reverse_sorting_order, ModelFinder, ModelNotSetError
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
    """Testing if returned string representation of object from name() function is correct."""
    actual_result = obj_name(obj)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_str", "expected_result"),
    (
            ("roc_auc_score", True),
            ("mean_squared_error", False),
            ("mean_negative_loss", False),
            ("loss_score", True),
            ("error", True),
            ("loss", True),
            ("error_loss", False),
            ("loss_error", False),
            ("test_string", True),
            ("qualityloss", True)
    )
)
def test_reverse_sorting_order(input_str, expected_result):
    """Testing if assessment of sorting order from reverse_sorting_order() is correct."""
    assert reverse_sorting_order(input_str) == expected_result


@pytest.mark.parametrize(
    ("category_type",),
    (
            ("categorical",),
            ("numerical",),
    )
)
def test_model_finder_init(data_classification_balanced, split_dataset_categorical, seed, category_type):
    """Testing if initialization of ModelFinder's properties works correctly depending on the category_type."""
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
    """Testing if error is raised when incorrect category_type is provided to ModelFinder."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    err_msg = "Expected one of the categories: "
    with pytest.raises(ValueError) as excinfo:
        ModelFinder(
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
    """Testing if DummyModel (for classification) is created correctly."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    expected_model = DummyClassifier(strategy="stratified", random_state=seed)
    expected_model.fit(X, y)
    expected_model_scores = {"roc_auc_score": 0.41666666666666663, "accuracy_score": 0.48}

    model_finder_classification.scoring_functions = [roc_auc_score, accuracy_score]
    actual_model, actual_model_scores = model_finder_classification._create_dummy_model()

    assert str(actual_model) == str(expected_model)
    assert np.array_equal(actual_model.predict(test_input), expected_model.predict(test_input))
    assert actual_model_scores == expected_model_scores


@pytest.mark.parametrize(
    ("test_input",),
    (
            ([0, 1, 2, 3, 4, 5],),
            ([6, 7, 8, 9, 10, 11],),
            ([12, 13, 14, 15, 16, 17],),
    )
)
def test_model_finder_dummy_regression(model_finder_regression, test_input):
    """Testing if DummyModel (for regression) is created correctly."""
    expected_model = DummyRegressor(strategy="median")
    median = 35.619966243279364
    expected_model_scores = {"mean_squared_error": 487.0142795860736, "r2_score": -0.003656622727187031}

    model_finder_regression.scoring_functions = [mean_squared_error, r2_score]
    actual_model, actual_model_scores = model_finder_regression._create_dummy_model()

    assert str(actual_model) == str(expected_model)
    assert np.array_equal(actual_model.predict(test_input), np.array([median] * len(test_input)))
    assert actual_model_scores == expected_model_scores


def test_model_finder_classification_dummy_model_results(model_finder_classification, seed):
    """Testing if dummy_model_results() function returns correct DataFrame (classification)."""
    _ = {
        "model": "DummyClassifier",
        "fit_time": np.nan,
        "params": "{{'constant': None, 'random_state': {seed}, 'strategy': 'stratified'}}".format(seed=seed),
        "roc_auc_score": 0.41666666666666663,
        "f1_score": 0.6285714285714286,
        "accuracy_score": 0.48,
        "balanced_accuracy_score": 0.41666666666666663
    }
    expected_df = pd.DataFrame(_, index=[9999])
    actual_df = model_finder_classification._dummy_model_results()

    assert actual_df.equals(expected_df[actual_df.columns])


def test_model_finder_regression_dummy_model_results(model_finder_regression):
    """Testing if dummy_model_results() function returns correct DataFrame (regression)."""
    _ = {
        "model": "DummyRegressor",
        "fit_time": np.nan,
        "params": "{'constant': None, 'quantile': None, 'strategy': 'median'}",
        "mean_squared_error": 487.0142795860736,
        "mean_absolute_error": 14.28810797425516,
        "explained_variance_score": 0.0,
        "r2_score": -0.003656622727187031
    }
    expected_df = pd.DataFrame(_, index=[9999])
    actual_df = model_finder_regression._dummy_model_results()

    assert actual_df.equals(expected_df[actual_df.columns])


@pytest.mark.parametrize(
    ("mode", "expected_model"),
    (
            ("quick", SVC(tol=0.1, C=0.1)),
            ("detailed", DecisionTreeClassifier(criterion="entropy", max_depth=10))
    )
)
def test_model_finder_classification_search(model_finder_classification, mode, expected_model, seed):
    """Testing if search() function returns expected Model (for classification)."""
    model_finder_classification._quicksearch_limit = 1
    actual_model = model_finder_classification.search(models=None, scoring=roc_auc_score, mode=mode)
    expected_model.random_state = seed
    assert str(actual_model) == str(expected_model)


@pytest.mark.parametrize(
    ("models", "expected_model"),
    (
            ([
                 RidgeClassifier(alpha=1.0, random_state=1),
                 RidgeClassifier(alpha=100.0, random_state=1)
             ],
             RidgeClassifier(alpha=100.0, random_state=1)),

            ([
                 SVC(C=1.0, random_state=10),
                 SVC(C=10.0, random_state=14),
                 SVC(C=100.0, random_state=35)
             ],
             SVC(C=1.0, random_state=10))
    )
)
def test_model_finder_classification_search_defined_models(model_finder_classification, models, expected_model):
    """Testing if models provided explicitly are being scored and chosen properly in classification
    (including models not present in default models collection."""
    actual_model = model_finder_classification.search(models=models, scoring=roc_auc_score)
    assert str(actual_model) == str(expected_model)


@pytest.mark.parametrize(
    ("mode", "expected_model"),
    (
            ("quick", SVR(C=0.1, tol=1.0)),
            ("detailed", SVR(C=0.1, tol=1.0))
    )
)
def test_model_finder_regression_search(model_finder_regression, mode, expected_model, seed):
    """Testing if search() function returns expected Model (for regression)."""
    model_finder_regression._quicksearch_limit = 1
    actual_model = model_finder_regression.search(models=None, scoring=mean_squared_error, mode=mode)
    expected_model.random_state = seed
    assert str(actual_model) == str(expected_model)


@pytest.mark.parametrize(
    ("models", "expected_model"),
    (
            ([
                 DecisionTreeRegressor(max_depth=10, criterion="mae", random_state=1),
                 DecisionTreeRegressor(max_depth=100, criterion="mse", random_state=1)
             ],
             DecisionTreeRegressor(max_depth=100, criterion="mse", random_state=1)),

            ([
                 LinearSVR(C=1.0),
                 LinearSVR(C=10.0),
                 LinearSVR(C=100.0)
             ],
             LinearSVR(C=1.0))
    )
)
def test_model_finder_regression_search_defined_models(model_finder_regression, models, expected_model):
    """Testing if models provided explicitly are being scored and chosen properly in regression
    (including models not present in default models collection."""
    actual_model = model_finder_regression.search(models=models, scoring=mean_squared_error)
    assert str(actual_model) == str(expected_model)


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
    """Testing if search() function raises an error when incorrect mode is provided."""
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
    """Testing if search() function raises an error when incorrect type of models is provided."""
    mode = model_finder_classification._mode_quick
    with pytest.raises(ValueError) as excinfo:
        model_finder_classification.search(models=incorrect_model, mode=mode)
    assert "models should be Dict, List-like or None" in str(excinfo.value)


def test_model_finder_fit(model_finder_classification, seed):
    """Testing if fit() function properly fits the model."""
    mf = model_finder_classification
    mf.set_model(LogisticRegression())
    mf.fit()
    try:
        mf.predict(mf.X.toarray())
    except NotFittedError:
        pytest.fail()


def test_model_finder_fit_no_model(model_finder_classification):
    """Testing if fit() function raises an error when there is no Model set."""
    with pytest.raises(ModelNotSetError):
        model_finder_classification.fit()


def test_model_finder_predict(model_finder_classification, seed):
    """Testing if predict() function correctly predicts the output."""
    expected_result = [1, ]
    test = np.array([1.34, -0.25, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]).reshape(1, -1)

    mf = model_finder_classification
    mf.set_model(LogisticRegression(random_state=seed))
    mf.fit()
    actual_result = mf.predict(test)

    assert actual_result == expected_result


def test_model_finder_predict_no_model(model_finder_classification):
    """Testing if predict() function raises an error when no Model is set."""
    with pytest.raises(ModelNotSetError):
        model_finder_classification.predict(["test_input"])


def test_model_finder_perform_gridsearch_classification(model_finder_classification, chosen_classifiers_grid, seed):
    """Testing if gridsearch works and returns correct Models and result dict (in classification)."""
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

    # checking if the keys in dicts from actual_results match what is expected
    assert len(actual_results.keys()) == len(expected_models)
    for model_tuple in expected_models:
        ml = model_tuple[0]
        params = model_tuple[1]
        ml_specific_keys = ["param_" + param for param in params.keys()] + standard_keys
        expected_keys = set(ml_specific_keys)
        actual_keys = set(actual_results[ml].keys())
        assert actual_keys == expected_keys


def test_model_finder_perform_gridsearch_regression(model_finder_regression, chosen_regressors_grid, seed):
    """Testing if gridsearch works and returns correct Models and result dict (in regression)."""
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
    # checking if the keys in dicts from actual_results match what is expected
    assert len(actual_results.keys()) == len(expected_models)
    for model_tuple in expected_models:
        ml = model_tuple[0]
        params = model_tuple[1]
        ml_specific_keys = ["param_" + param for param in params.keys()] + standard_keys
        expected_keys = set(ml_specific_keys)
        actual_keys = set(actual_results[ml].keys())
        assert actual_keys == expected_keys


@pytest.mark.parametrize(
    ("input_dict", "expected_result"),
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
                 pd.DataFrame: {"name": ["df1", "df2"], "reason": ["eggs", "eggs2"],
                                "score": [4, -1], "fit-time": [0, np.nan]}
             },
             pd.DataFrame(data={
                 "model": ["Exception", "Ridge", "Ridge", "Ridge", "DataFrame", "DataFrame"],
                 "name": ["exception", np.nan, np.nan, np.nan, "df1", "df2"],
                 "reason": ["none", np.nan, np.nan, np.nan, "eggs", "eggs2"],
                 "score": [np.nan, 3, 4, 5, 4, -1],
                 "fit-time": [np.nan, 1, 1, 4, 0, np.nan],
                 "test": ["done", None, None, None, np.nan, np.nan]
             })
            )
    )
)
def test_model_finder_create_gridsearch_results_dataframe(model_finder_classification, input_dict, expected_result):
    """Testing if creating gridsearch results dataframe works correctly."""
    expected_result = expected_result.rename({"model": model_finder_classification._model_name})
    actual_result = model_finder_classification._create_gridsearch_results_dataframe(input_dict)

    assert actual_result.equals(expected_result[actual_result.columns])


def test_model_finder_perform_quicksearch_classification(model_finder_classification, chosen_classifiers_grid, seed):
    """Testing if quicksearch works and returns correct Models and result dict (in classification)."""
    expected_models = [
        (DecisionTreeClassifier, 0.5773809523809523),
        (LogisticRegression, 0.6071428571428571),
        (SVC, 0.6309523809523809),
    ]
    expected_keys = {"fit_time", "roc_auc_score", "params"}

    actual_models, actual_results = model_finder_classification._perform_quicksearch(
        chosen_classifiers_grid, roc_auc_score
    )

    assert sorted(actual_models, key=lambda x: x[0].__name__) == expected_models
    # checking if the keys in dicts from actual_results match what is expected
    assert len(actual_results.keys()) == len(expected_models)
    for model_tuple in expected_models:
        model = model_tuple[0]
        actual_keys = set(actual_results[model].keys())
        assert actual_keys == expected_keys


def test_model_finder_perform_quicksearch_regression(model_finder_regression, chosen_regressors_grid, seed):
    """Testing if quicksearch works and returns correct Models and result dict (in regression)."""
    expected_models = [
        (DecisionTreeRegressor, 2458.2551351805055),
        (Ridge, 1199.0610634709196),
        (SVR, 1051.8445189635734),
    ]
    expected_keys = {"fit_time", "mean_squared_error", "params"}

    actual_models, actual_results = model_finder_regression._perform_quicksearch(
        chosen_regressors_grid, mean_squared_error
    )

    assert sorted(actual_models, key=lambda x: x[0].__name__) == expected_models
    # checking if the keys in dicts from actual_results match what is expected
    assert len(actual_results.keys()) == len(expected_models)
    for model_tuple in expected_models:
        model = model_tuple[0]
        actual_keys = set(actual_results[model].keys())
        assert actual_keys == expected_keys


@pytest.mark.parametrize(
    ("input_dict", "scoring", "expected_result"),
    (
            ({
                 Ridge: {"test": 1, "roc_auc_score": 12, "params": "abcd"},
                 LogisticRegression: {"test": 1, "roc_auc_score": 10, "params": "abcd"},
                 DecisionTreeClassifier: {"test": 2, "roc_auc_score": 15, "params": "xyz"}
             },
             roc_auc_score,
             pd.DataFrame(
                 data={
                     "model": ["DecisionTreeClassifier", "Ridge", "LogisticRegression"],
                     "test": [2, 1, 1],
                     "roc_auc_score": [15, 12, 10],
                     "params": ["xyz", "abcd", "abcd"]
                 },
                 index=[2, 0, 1]
             )
            ),

            ({
                 Exception: {"mean_squared_error": 8, "fit_time": 10},
                 Ridge: {"mean_squared_error": 10, "fit_time": 15},
                 LogisticRegression: {"mean_squared_error": 4, "fit_time": "test"}
             },
             mean_squared_error,
             pd.DataFrame(
                 data={
                     "model": ["LogisticRegression", "Exception", "Ridge"],
                     "mean_squared_error": [4, 8, 10],
                     "fit_time": ["test", 10, 15]
                 },
                 index=[2, 0, 1]
             )
            )
    )
)
def test_model_finder_create_search_results_dataframe(model_finder_classification, input_dict, scoring,
                                                      expected_result):
    """Testing if creating search results dataframe works correctly."""
    actual_result = model_finder_classification._create_search_results_dataframe(input_dict, scoring)
    assert actual_result.equals(expected_result[actual_result.columns])


@pytest.mark.parametrize(
    ("limit", "expected_models"),
    (
            (1, [SVC]),
            (2, [SVC, LogisticRegression])
    )
)
def test_model_finder_quicksearch_classification(
        model_finder_classification, chosen_classifiers_grid, limit, expected_models
):
    """Testing if quicksearch correctly chooses only a limited number of found Models based on the limit
    (in classification)."""
    model_finder_classification._quicksearch_limit = limit
    actual_models = model_finder_classification._quicksearch(chosen_classifiers_grid, roc_auc_score)

    assert actual_models == expected_models


@pytest.mark.parametrize(
    ("limit", "expected_models"),
    (
            (1, [SVR]),
            (2, [SVR, Ridge])
    )
)
def test_model_finder_quicksearch_regression(
        model_finder_regression, chosen_regressors_grid, limit, expected_models
):
    """Testing if quicksearch correctly chooses only a limited number of found Models based on the limit
    (in regression)."""
    model_finder_regression._quicksearch_limit = limit
    actual_models = model_finder_regression._quicksearch(chosen_regressors_grid, mean_squared_error)

    assert actual_models == expected_models


def test_model_finder_assess_models_classification(model_finder_classification, seed):
    """Testing if assess_model function returns correct Models and result dict (in classification)."""
    models = [
        DecisionTreeClassifier(**{"max_depth": 10, "criterion": "entropy", "random_state": seed}),
        LogisticRegression(**{"C": 1.0, "tol": 0.1, "random_state": seed}),
        SVC(**{"C": 0.1, "tol": 0.1, "random_state": seed})
    ]
    scores = [0.5333333333333333, 0.43333333333333335, 0.5]

    expected_models = list(zip(models, scores))
    expected_keys = {"fit_time", "roc_auc_score", "params", "accuracy_score", "balanced_accuracy_score", "f1_score"}

    actual_models, actual_results = model_finder_classification._assess_models(models, roc_auc_score)

    assert actual_models == expected_models
    assert len(actual_results.keys()) == len(expected_models)

    for model in actual_results:
        assert set(actual_results[model].keys()) == set(expected_keys)  # testing keys from act and exp dicts


def test_model_finder_assess_models_regression(model_finder_regression, seed):
    """Testing if assess_model function returns correct Models and result dict (in regression)."""
    models = [
        DecisionTreeRegressor(**{"max_depth": 10, "criterion": "mae", "random_state": seed}),
        Ridge(**{"alpha": 0.0001, "random_state": seed}),
        SVR(**{"C": 0.1, "tol": 1.0})
    ]
    scores = [1323.0223273506956, 628.4920219435661, 486.38607353926875]

    expected_models = list(zip(models, scores))
    expected_keys = {
        "fit_time", "mean_squared_error", "params", "mean_absolute_error", "explained_variance_score", "r2_score"
    }

    actual_models, actual_results = model_finder_regression._assess_models(models, mean_squared_error)

    assert actual_models == expected_models
    assert len(actual_results.keys()) == len(expected_models)

    for model in actual_results:
        assert set(actual_results[model].keys()) == set(expected_keys)  # testing keys from act and exp dicts


@pytest.mark.parametrize(
    ("scoring", "expected_result"),
    (
            (roc_auc_score, [roc_auc_score, r2_score, accuracy_score]),
            (r2_score, [r2_score, accuracy_score]),
            (mean_squared_error, [mean_squared_error, r2_score, accuracy_score]),
            (accuracy_score, [r2_score, accuracy_score]),
    )
)
def test_model_finder_get_scorings(model_finder_classification, scoring, expected_result):
    """Testing if appending the scoring to default scoring_functions works properly."""
    model_finder_classification.scoring_functions = [r2_score, accuracy_score]
    actual_result = model_finder_classification._get_scorings(scoring)

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("model", "expected_results"),
    (
            (LogisticRegression(C=1.0, tol=0.1, random_state=1010),
             {"roc_auc_score": 0.43333333333333335, "accuracy_score": 0.52}),
            (LogisticRegression(C=10.0, tol=1, random_state=42), {"roc_auc_score": 0.4, "accuracy_score": 0.48}),
            (DecisionTreeClassifier(max_depth=100, criterion="gini", random_state=42),
             {"roc_auc_score": 0.5333333333333333, "accuracy_score": 0.6}),
            (SVC(C=1.0, tol=0.1, random_state=1), {"roc_auc_score": 0.5166666666666667, "accuracy_score": 0.6})
    )
)
def test_model_finder_score_model(model_finder_classification, model, expected_results):
    """Testing if score_model() function properly scores different Models."""
    X_train, y_train = model_finder_classification.X_train, model_finder_classification.y_train
    model.fit(X_train, y_train)
    model_finder_classification.scoring_functions = [roc_auc_score, accuracy_score]
    actual_results = model_finder_classification._score_model(model, roc_auc_score)

    assert actual_results == expected_results


def test_model_finder_set_model(model_finder_classification, seed):
    """Testing if set_model() function correctly sets chosen Model and corresponding properties.
    Additionally checks if the set Model wasn't fitter in the process."""
    model = LogisticRegression(C=1.0, tol=0.1, random_state=seed)
    mf = model_finder_classification
    mf.scoring_functions = [roc_auc_score, accuracy_score]
    mf.set_model(model)

    assert mf._chosen_model == model
    assert mf._chosen_model_params == model.get_params()
    assert mf._chosen_model_scores == {"roc_auc_score": 0.43333333333333335, "accuracy_score": 0.52}

    with pytest.raises(NotFittedError):
        mf.predict([1])


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
    )
)
def test_model_finder_classification_search_results_dataframe(model_finder_classification_fitted, limit, seed):
    models = ["DecisionTreeClassifier", "SVC", "LogisticRegression"]
    dummy = ["DummyClassifier"]
    expected_index = models[:limit] + dummy
    expected_keys = {"fit_time", "params", "roc_auc_score", "accuracy_score", "balanced_accuracy_score", "f1_score"}

    actual_results = model_finder_classification_fitted.search_results(limit)

    assert actual_results.index.tolist() == expected_index
    assert set(actual_results.columns) == expected_keys


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
    )
)
def test_model_finder_regression_search_results_dataframe(model_finder_regression_fitted, limit, seed):
    models = ["SVR", "Ridge", "DecisionTreeRegressor"]
    dummy = ["DummyRegressor"]
    expected_index = models[:limit] + dummy
    expected_keys = {
        "fit_time", "params", "mean_squared_error", "r2_score", "mean_absolute_error", "explained_variance_score"
    }

    actual_results = model_finder_regression_fitted.search_results(limit)

    assert actual_results.index.tolist() == expected_index
    assert set(actual_results.columns) == expected_keys


def test_model_finder_roc_curves(model_finder_classification_fitted):
    curves = model_finder_classification_fitted.roc_curves(4)
    for model, curve in curves:
        print(model)
        print(curve)

    assert False