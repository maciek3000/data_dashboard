import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, det_curve
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import NotFittedError

from data_dashboard.model_finder import ModelsNotSearchedError


@pytest.mark.parametrize(
    ("test_input",),
    (
            ([0, 1, 2, 3, 4, 5],),
            ([6, 7, 8, 9, 10, 11],),
            ([12, 13, 14, 15, 16, 17],),
    )
)
def test_model_finder_dummy_classification(model_finder_classification, split_dataset_categorical, seed, test_input):
    """Testing if DummyModel (for classification) is created correctly."""
    X_train = split_dataset_categorical[0]
    y_train = split_dataset_categorical[2]
    expected_model = DummyClassifier(strategy="stratified", random_state=seed)
    expected_model.fit(X_train, y_train)
    expected_model_scores = {"roc_auc_score": 0.41666666666666663, "accuracy_score": 0.48}

    model_finder_classification.scoring_functions = [roc_auc_score, accuracy_score]
    actual_model, actual_model_scores = model_finder_classification._create_dummy_model()

    assert str(actual_model) == str(expected_model)
    assert np.array_equal(actual_model.predict(test_input), expected_model.predict(test_input))
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


def test_model_finder_set_model_classification(model_finder_classification, seed):
    """Testing if set_model() function correctly sets chosen Model and corresponding properties (classification).
    Additionally checks if the set Model wasn't fitted in the process."""
    model = LogisticRegression(C=1.0, tol=0.1, random_state=seed)
    mf = model_finder_classification
    mf.scoring_functions = [roc_auc_score, accuracy_score]
    mf.set_model(model)

    assert mf._chosen_model == model
    assert mf._chosen_model_params == model.get_params()
    assert mf._chosen_model_scores == {"roc_auc_score": 0.6666666666666667, "accuracy_score": 0.52}

    with pytest.raises(NotFittedError):
        mf.predict([1])


@pytest.mark.parametrize(
    ("mode", "expected_model"),
    (
            ("quick", SVC(tol=0.1, C=0.1)),
            ("detailed", LogisticRegression(tol=0.1))
    )
)
def test_model_finder_classification_search(model_finder_classification, mode, expected_model, seed):
    """Testing if search() function returns expected Model (for classification)."""
    model_finder_classification._quicksearch_limit = 1
    actual_model = model_finder_classification.search(models=None, scoring=roc_auc_score, mode=mode)
    expected_model.random_state = seed
    assert str(actual_model) == str(expected_model)


@pytest.mark.parametrize(
    ("mode", "expected_model", "expected_scores"),
    (
            ("quick", SVC(tol=0.1, C=0.1), {"roc_auc_score": 0.6066666666666667, "accuracy_score": 0.6}),
            ("detailed", LogisticRegression(tol=0.1), {"roc_auc_score": 0.6666666666666667, "accuracy_score": 0.52})
    )
)
def test_model_finder_search_and_fit_classification(model_finder_classification, mode, expected_model, expected_scores, seed):
    """Testing if search_and_fit() function correctly searches for and sets and fits chosen model (classification)."""
    prediction_array = np.array([1.34, -0.25, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]).reshape(1, -1)
    model_finder_classification._quicksearch_limit = 1
    model_finder_classification.scoring_functions = [roc_auc_score, accuracy_score]
    actual_model = model_finder_classification.search_and_fit(models=None, scoring=roc_auc_score, mode=mode)
    expected_model.random_state = seed

    assert str(actual_model) == str(expected_model)
    assert str(model_finder_classification._chosen_model) == str(expected_model)
    assert model_finder_classification._chosen_model_params == expected_model.get_params()
    assert model_finder_classification._chosen_model_scores == expected_scores

    assert model_finder_classification.predict(prediction_array) == [1]


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
    (including models not present in default models collection)."""
    actual_model = model_finder_classification.search(models=models, scoring=roc_auc_score)
    assert str(actual_model) == str(expected_model)


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


def test_model_finder_perform_quicksearch_classification(model_finder_classification, chosen_classifiers_grid, seed):
    """Testing if quicksearch works and returns correct Models and result dict (in classification)."""
    expected_models = [
        (DecisionTreeClassifier, 0.5773809523809523),
        (LogisticRegression, 0.6428571428571429),
        (SVC, 0.7261904761904763),
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


def test_model_finder_assess_models_classification(model_finder_classification, seed):
    """Testing if assess_model function returns correct Models and result dict (in classification)."""
    models = [
        DecisionTreeClassifier(**{"max_depth": 10, "criterion": "entropy", "random_state": seed}),
        LogisticRegression(**{"tol": 0.1, "random_state": seed}),
        SVC(**{"C": 0.1, "tol": 0.1, "random_state": seed})
    ]
    scores = [0.5333333333333333, 0.6666666666666667, 0.6066666666666667]

    expected_models = list(zip(models, scores))
    expected_keys = {"fit_time", "roc_auc_score", "params", "accuracy_score", "balanced_accuracy_score", "f1_score"}

    actual_models, actual_results = model_finder_classification._assess_models_performance(models, roc_auc_score)

    assert actual_models == expected_models
    assert len(actual_results.keys()) == len(expected_models)

    for model in actual_results:
        assert set(actual_results[model].keys()) == set(expected_keys)  # testing keys from act and exp dicts


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
    )
)
def test_model_finder_classification_search_results_dataframe(model_finder_classification_fitted, limit, seed):
    """Testing if search_results_dataframe is being correctly filtered out to a provided
    model_limit (in classification)"""
    models = ["LogisticRegression", "SVC", "DecisionTreeClassifier"]
    dummy = ["DummyClassifier"]
    expected_index = models[:limit] + dummy
    expected_keys = {"fit_time", "params", "roc_auc_score", "accuracy_score", "balanced_accuracy_score", "f1_score"}

    actual_results = model_finder_classification_fitted.search_results(limit)

    assert actual_results.index.tolist() == expected_index
    assert set(actual_results.columns) == expected_keys


@pytest.mark.parametrize(
    ("model", "params", "response_method", "plot_func"),
    (
            (LogisticRegression, {"C": 1.0, "tol": 0.1}, "predict_proba", roc_curve),
            (SVC, {"C": 100.0}, "decision_function", roc_curve),
            (LogisticRegression, {"C": 100.0, "tol": 10.0}, "predict_proba", det_curve),
            (SVC, {"C": 1.0, "tol": 100.0}, "decision_function", det_curve)
    )
)
def test_model_finder_classification_plot_curves(
        model_finder_classification_fitted, split_dataset_categorical, seed, model, params, response_method, plot_func
):
    """Testing if _plot_curves correctly assesses prediction probabilities and calculates the results based on the
    provided plot_func."""

    params["random_state"] = seed
    clf = model(**params)
    X_train, X_test, y_train, y_test = split_dataset_categorical
    clf.fit(X_train, y_train)
    y_scores = getattr(clf, response_method)(X_test)

    if response_method == "predict_proba":
        y_scores = y_scores[:, 1]

    expected_results = plot_func(y_test, y_scores)

    mock_search_results = [(clf,), ]
    model_finder_classification_fitted._search_results = mock_search_results
    limit = None
    actual_results = model_finder_classification_fitted._plot_curves(plot_func, limit)[0][1]

    for actual_arr, expected_arr in zip(actual_results, expected_results):
        assert np.array_equal(actual_arr, expected_arr)


def test_model_finder_classification_plot_curves_error(model_finder_classification):
    """Testing if _plot_curves raises an Exception when there are no search results available (classification)."""
    with pytest.raises(ModelsNotSearchedError) as excinfo:
        model_finder_classification._plot_curves("test_func", 1)
    assert "Search Results is not available. " in str(excinfo.value)


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_classification_confusion_matrices(model_finder_classification_fitted, limit):
    """Testing if confusion matrices are being correctly calculated and returned (in classification)."""
    results = [
        ("LogisticRegression", [0, 10, 2, 13]),
        ("SVC", [0, 10, 0, 15]),
        ("DecisionTreeClassifier", [2, 8, 2, 13])
    ]
    expected_results = results[:limit]

    actual_results = model_finder_classification_fitted.confusion_matrices(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert actual_result[0].__class__.__name__ == expected_result[0]
        assert actual_result[1].shape == (2, 2)
        assert actual_result[1].ravel().tolist() == expected_result[1]


def test_model_finder_classification_confusion_matrices_error(model_finder_classification):
    """Testing if confusion_matrices raises an error when there are no search results available (classification)."""
    with pytest.raises(ModelsNotSearchedError) as excinfo:
        _ = model_finder_classification.confusion_matrices(1)
    assert "Search Results is not available. " in str(excinfo.value)


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_predict_X_test_classification(
        model_finder_classification_fitted, split_dataset_categorical, limit, seed
):
    """Testing if predictions of X_test split from found models are correct (in classification)."""
    models = [
        LogisticRegression(**{"tol": 0.1, "random_state": seed}),
        SVC(**{"C": 0.1, "tol": 0.1, "random_state": seed}),
        DecisionTreeClassifier(**{"max_depth": 10, "criterion": "entropy", "random_state": seed})
    ]
    results = []
    X_train, X_test, y_train, y_test = split_dataset_categorical
    for model in models:
        new_model = model.fit(X_train, y_train)
        results.append((model, new_model.predict(X_test)))

    expected_results = results[:limit]

    actual_results = model_finder_classification_fitted.predictions_X_test(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert str(actual_result[0]) == str(expected_result[0])
        assert np.array_equal(actual_result[1], expected_result[1])


@pytest.mark.parametrize(
    ("test_model",),
    (
            (LogisticRegression(),),
            (DecisionTreeClassifier(max_depth=5),),
            (SVC(C=1000),)
    )
)
def test_model_finder_wrap_model_classification(model_finder_classification, test_model):
    """Testing if wrapping Model in classification doesn't change input variable."""
    actual_model = model_finder_classification._wrap_model(test_model)
    assert actual_model == test_model


@pytest.mark.parametrize(
    ("test_data",),
    (
            ({"A": [1, 2, 3], "B": [10, 20, 30]},),
            ({"aa": [1, 2], "B": [1, 2], "C": [3, 3], "last": [4, 100]},)
    )
)
def test_model_finder_wrap_results_dataframe_classification(model_finder_classification, test_data):
    """Testing if wrapping DataFrame in classification doesn't change input variable."""
    expected_df = pd.DataFrame(data=test_data)
    actual_df = model_finder_classification._wrap_results_dataframe(expected_df)

    assert actual_df.equals(expected_df)


@pytest.mark.parametrize(
    ("test_params",),
    (
            ({"A": 100, "B": "test", "C": [1, 2]},),
            ({"uno": 1, "dos": 2, "tres": "quatro"},)
    )
)
def test_model_finder_wrap_params_classification(model_finder_classification, test_params):
    """Testing if wrapping params in classification doesn't change input variable."""
    expected_params = test_params
    actual_params = model_finder_classification._wrap_params(test_params)
    assert actual_params == expected_params


@pytest.mark.parametrize(
    ("model",),
    (
            (LogisticRegression(),),
            (SVC(C=1000.0),),
            (DecisionTreeClassifier(max_depth=10, criterion="entropy"),)
    )
)
def test_model_finder_calculate_model_score_classification_regular_scoring(model_finder_classification, split_dataset_categorical, model):
    """Testing if calculating model score works correctly in classification with scoring != roc_auc_score."""
    scoring = accuracy_score
    X_train = split_dataset_categorical[0]
    X_test = split_dataset_categorical[1]
    y_train = split_dataset_categorical[2]
    y_test = split_dataset_categorical[3]

    model.fit(X_train, y_train)

    expected_result = scoring(y_test, model.predict(X_test))
    actual_result = model_finder_classification._calculate_model_score(model, X_test, y_test, scoring)

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("model",),
    (
            (LogisticRegression(),),
            (DecisionTreeClassifier(max_depth=10, criterion="entropy"),),
    )
)
def test_model_finder_calculate_model_score_classification_roc_auc_scoring_proba(model_finder_classification, split_dataset_categorical, model):
    """Testing if calculating model score works correctly in classification with scoring == roc_auc_score
    and with models exposing predict_proba() method."""
    scoring = roc_auc_score
    X_train = split_dataset_categorical[0]
    X_test = split_dataset_categorical[1]
    y_train = split_dataset_categorical[2]
    y_test = split_dataset_categorical[3]

    model.fit(X_train, y_train)

    expected_result = scoring(y_test, model.predict_proba(X_test)[:, 1])
    actual_result = model_finder_classification._calculate_model_score(model, X_test, y_test, scoring)

    assert actual_result, expected_result

@pytest.mark.parametrize(
    ("model",),
    (
            (SVC(C=1000.0),),
            (PassiveAggressiveClassifier(),),
    )
)
def test_model_finder_calculate_model_score_classification_roc_auc_scoring_decision_func(model_finder_classification, split_dataset_categorical, model):
    """Testing if calculating model score works correctly in classification with scoring == roc_auc_score
    and with models exposing decision_function() method."""
    scoring = roc_auc_score
    X_train = split_dataset_categorical[0]
    X_test = split_dataset_categorical[1]
    y_train = split_dataset_categorical[2]
    y_test = split_dataset_categorical[3]

    model.fit(X_train, y_train)

    expected_result = scoring(y_test, model.decision_function(X_test))
    actual_result = model_finder_classification._calculate_model_score(model, X_test, y_test, scoring)

    assert actual_result, expected_result
