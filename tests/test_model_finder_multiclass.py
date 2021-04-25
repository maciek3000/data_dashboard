import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.dummy import DummyClassifier


@pytest.mark.parametrize(
    ("test_input",),
    (
            ([0, 1, 2, 3, 4, 5],),
            ([6, 7, 8, 9, 10, 11],),
            ([12, 13, 14, 15, 16, 17],),
    )
)
def test_model_finder_dummy_multiclass(
        model_finder_multiclass, split_dataset_multiclass, seed, multiclass_scorings, test_input
):
    """Testing if DummyModel (for multiclass) is created correctly."""
    X_train = split_dataset_multiclass[0]
    y_train = split_dataset_multiclass[2]
    expected_model = DummyClassifier(strategy="stratified", random_state=seed)
    expected_model.fit(X_train, y_train)
    expected_model_scores = {"f1_score_weighted": 0.4772161172161172, "precision_score_weighted": 0.508}

    model_finder_multiclass.scoring_functions = multiclass_scorings
    model_finder_multiclass.default_scoring = multiclass_scorings[0]
    actual_model, actual_model_scores = model_finder_multiclass._create_dummy_model()

    print(expected_model.predict(test_input))

    assert str(actual_model) == str(expected_model)
    assert np.array_equal(actual_model.predict(test_input), expected_model.predict(test_input))
    assert actual_model_scores == expected_model_scores


def test_model_finder_multiclass_dummy_model_results(model_finder_multiclass, seed):
    """Testing if dummy_model_results() function returns correct DataFrame (multiclass)."""
    _ = {
        "model": "DummyClassifier",
        "fit_time": np.nan,
        "params": "{{'constant': None, 'random_state': {seed}, 'strategy': 'stratified'}}".format(seed=seed),
        "f1_score_micro": 0.48,
        "f1_score_weighted": 0.4772161172161172,
        "precision_score_weighted": 0.508,
        "recall_score_weighted": 0.48,
        "accuracy_score": 0.48,
        "balanced_accuracy_score": 0.4987373737373737
    }
    expected_df = pd.DataFrame(_, index=[9999])
    actual_df = model_finder_multiclass._dummy_model_results()

    assert actual_df.equals(expected_df[actual_df.columns])


@pytest.mark.parametrize(
    ("mode", "expected_model"),
    (
            ("quick", DecisionTreeClassifier(max_depth=10)),
            ("detailed", LogisticRegression(tol=0.1))
    )
)
def test_model_finder_multiclass_search(model_finder_multiclass, multiclass_scorings, mode, expected_model, seed):
    """Testing if search() function returns expected Model (for multiclass)."""
    model_finder_multiclass._quicksearch_limit = 1
    actual_model = model_finder_multiclass.search(models=None, scoring=multiclass_scorings[0], mode=mode)
    expected_model.random_state = seed
    assert str(actual_model) == str(expected_model)


@pytest.mark.parametrize(
    ("models", "expected_model"),
    (
            ([
                 RidgeClassifier(alpha=1.0, random_state=1),
                 RidgeClassifier(alpha=100.0, random_state=1)
             ],
             RidgeClassifier(alpha=1.0, random_state=1)),

            ([
                 SVC(C=1.0, random_state=10),
                 SVC(C=10.0, random_state=14),
                 SVC(C=100.0, random_state=35)
             ],
             SVC(C=10.0, random_state=14))
    )
)
def test_model_finder_multiclass_search_defined_models(
        model_finder_multiclass, multiclass_scorings, models, expected_model
):
    """Testing if models provided explicitly are being scored and chosen properly in multiclass
    (including models not present in default models collection)."""
    actual_model = model_finder_multiclass.search(models=models, scoring=multiclass_scorings[0])
    assert str(actual_model) == str(expected_model)


def test_model_finder_perform_gridsearch_multiclass(
        model_finder_multiclass, multiclass_scorings, chosen_classifiers_grid, seed
):
    """Testing if gridsearch works and returns correct Models and result dict (in multiclass)."""
    expected_models = [
        (DecisionTreeClassifier, {"max_depth": 10, "criterion": "gini", "random_state": seed}),
        (LogisticRegression, {"C": 1.0, "tol": 0.1, "random_state": seed}),
        (SVC, {"C": 1.0, "tol": 0.1, "random_state": seed})
    ]
    standard_keys = [
        "iter", "n_resources", "mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "params",
        "split0_train_score", "split1_train_score", "split2_train_score", "split3_train_score", "split4_train_score",
        "split0_test_score", "split1_test_score", "split2_test_score", "split3_test_score", "split4_test_score",
        "rank_test_score", "mean_test_score", "mean_train_score", "std_test_score", "std_train_score"
    ]

    actual_models, actual_results = model_finder_multiclass._perform_gridsearch(
        chosen_classifiers_grid, multiclass_scorings[0], cv=5
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


def test_model_finder_perform_quicksearch_multiclass(
        model_finder_multiclass, multiclass_scorings, chosen_classifiers_grid, seed
):
    """Testing if quicksearch works and returns correct Models and result dict (in multiclass)."""
    expected_models = [
        (DecisionTreeClassifier, 0.948507632718159),
        (LogisticRegression, 0.8982456140350877),
        (SVC, 0.6207827260458838),
    ]
    expected_keys = {"fit_time", "f1_score_weighted", "params"}

    actual_models, actual_results = model_finder_multiclass._perform_quicksearch(
        chosen_classifiers_grid, multiclass_scorings[0]
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
            (1, [DecisionTreeClassifier]),
            (2, [DecisionTreeClassifier, LogisticRegression])
    )
)
def test_model_finder_quicksearch_multiclass(
        model_finder_multiclass, chosen_classifiers_grid, multiclass_scorings, limit, expected_models
):
    """Testing if quicksearch correctly chooses only a limited number of found Models based on the limit
    (in multiclass)."""
    model_finder_multiclass._quicksearch_limit = limit
    actual_models = model_finder_multiclass._quicksearch(chosen_classifiers_grid, multiclass_scorings[0])

    assert actual_models == expected_models


def test_model_finder_assess_models_multiclass(model_finder_multiclass, multiclass_scorings, seed):
    """Testing if assess_model function returns correct Models and result dict (in multiclass)."""
    models = [
        DecisionTreeClassifier(**{"max_depth": 10, "criterion": "entropy", "random_state": seed}),
        LogisticRegression(**{"C": 1.0, "tol": 0.1, "random_state": seed}),
        SVC(**{"C": 0.1, "tol": 0.1, "random_state": seed})
    ]
    scores = [1.0, 1.0, 0.26888888888888896]

    expected_models = list(zip(models, scores))
    expected_keys = {"fit_time", "f1_score_weighted", "params", "precision_score_weighted"}

    actual_models, actual_results = model_finder_multiclass._assess_models_performance(models, multiclass_scorings[0])

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
def test_model_finder_multiclass_search_results_dataframe(model_finder_multiclass_fitted, limit, seed):
    """Testing if search_results_dataframe is being correctly filtered out to a provided
    model_limit (in multiclass)"""
    models = ["DecisionTreeClassifier", "LogisticRegression", "SVC"]
    dummy = ["DummyClassifier"]
    expected_index = models[:limit] + dummy
    expected_keys = {"fit_time", "params", "f1_score_weighted", "f1_score_micro", "precision_score_weighted",
                     "accuracy_score", "recall_score_weighted", "balanced_accuracy_score"}

    actual_results = model_finder_multiclass_fitted.search_results(limit)

    assert actual_results.index.tolist() == expected_index
    assert set(actual_results.columns) == expected_keys


@pytest.mark.parametrize(
    ("input_func", "expected_results"),
    (
            ((
                     lambda y_true, y_score, param_one, param_two: y_true + y_score + param_one + param_two,
                     {"param_one": 10, "param_two": 20}, "test_func1"
             ), [33, 4, 5]),
            (
                    (lambda y_true, y_score, param_multiply: (y_true + y_score) * param_multiply,
                     {"param_multiply": 100}, "test_func2"),
                    [300, 4, 5]
            )
    )
)
def test_model_finder_multiclass_create_scoring_multiclass(
        model_finder_multiclass, input_func, expected_results
):
    """Testing if creating closures (and adding them to regular scorings) for multiclass scorings works correctly."""
    y_actual = 1
    y_predicted = 2

    def plus_one(y_true, y_score):
        return y_true + y_score + 1

    def plus_two(y_true, y_score):
        return y_true + y_score + 2

    model_finder_multiclass._scoring_multiclass = [plus_one, plus_two]
    model_finder_multiclass._scoring_multiclass_parametrized = [input_func]

    scorings = model_finder_multiclass._create_scoring_multiclass()
    actual_results = []
    for sc in scorings:
        actual_results.append(sc(y_actual, y_predicted))

    assert actual_results == expected_results


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_multiclass_confusion_matrices(model_finder_multiclass_fitted, limit):
    """Testing if confusion matrices are being correctly calculated and returned (in multiclass)."""
    results = [
        ("DecisionTreeClassifier", [11, 0, 0, 0, 6, 0, 0, 0, 8]),
        ("LogisticRegression", [11, 0, 0, 0, 6, 0, 0, 0, 8]),
        ("SVC", [11, 0, 0, 0, 6, 0, 3, 0, 5])
    ]
    expected_results = results[:limit]
    actual_results = model_finder_multiclass_fitted.confusion_matrices(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert actual_result[0].__class__.__name__ == expected_result[0]
        assert actual_result[1].shape == (3, 3)
        assert actual_result[1].ravel().tolist() == expected_result[1]


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_predict_X_test_multiclass(model_finder_multiclass_fitted, split_dataset_multiclass, limit, seed):
    """Testing if predictions of X_test split from found models are correct (in multiclass)."""
    models = [
        DecisionTreeClassifier(**{"max_depth": 10, "random_state": seed}),
        LogisticRegression(**{"C": 1.0, "tol": 0.1, "random_state": seed}),
        SVC(**{"tol": 0.1, "random_state": seed})
    ]
    results = []
    X_train, X_test, y_train, y_test = split_dataset_multiclass
    for model in models:
        new_model = model.fit(X_train, y_train)
        results.append((model, new_model.predict(X_test)))

    expected_results = results[:limit]

    actual_results = model_finder_multiclass_fitted.predictions_X_test(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert str(actual_result[0]) == str(expected_result[0])
        assert np.array_equal(actual_result[1], expected_result[1])


@pytest.mark.parametrize(
    ("model",),
    (
            (LogisticRegression(),),
            (SVC(C=1000.0),),
            (DecisionTreeClassifier(max_depth=10, criterion="entropy"),)
    )
)
def test_model_finder_calculate_model_score_multiclass_regular_scoring(model_finder_multiclass, split_dataset_multiclass, model):
    """Testing if calculating model score works correctly in multiclass with scoring != roc_auc_score."""
    scoring = accuracy_score
    X_train = split_dataset_multiclass[0]
    X_test = split_dataset_multiclass[1]
    y_train = split_dataset_multiclass[2]
    y_test = split_dataset_multiclass[3]

    model.fit(X_train, y_train)

    expected_result = scoring(y_test, model.predict(X_test))
    actual_result = model_finder_multiclass._calculate_model_score(model, X_test, y_test, scoring)

    assert actual_result == expected_result
