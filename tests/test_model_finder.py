import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.exceptions import NotFittedError

from ml_dashboard.model_finder import ModelFinder, ModelNotSetError
from ml_dashboard.model_finder import WrappedModelRegression
from ml_dashboard.models import classifiers, regressors



@pytest.mark.parametrize(
    ("test_model",),
    (
            (Ridge(),),
            (LogisticRegression(),),
            (SVC(),),
            (DecisionTreeClassifier(),),
    )
)
def test_wrapped_model_regression(test_model, seed):
    """Testing if Regression Wrapper properly remaps properties and functions to those of the provided regressor."""
    wrapped = WrappedModelRegression(
        regressor=test_model,
        transformer=QuantileTransformer(output_distribution="normal", random_state=seed)
    )

    assert wrapped.__class__ == test_model.__class__
    assert wrapped.__name__ == test_model.__class__.__name__
    assert str(wrapped) == str(test_model)
    assert isinstance(wrapped.__class__(), test_model.__class__)

    assert type(wrapped) == WrappedModelRegression


@pytest.mark.parametrize(
    ("test_model",),
    (
            (Ridge(alpha=100),),
            (LogisticRegression(C=100, tol=0.01),),
            (SVC(C=1000),),
            (DecisionTreeClassifier(max_depth=10, criterion="entropy"),),
    )
)
def test_wrapped_model_regression_params(test_model, seed):
    """Testing if Regression Wrapper properly remaps get_params() function to that of the regressor."""
    wrapped = WrappedModelRegression(
        regressor=test_model,
        transformer=QuantileTransformer(output_distribution="normal", random_state=seed)
    )
    assert wrapped.get_params() == test_model.get_params()


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


@pytest.mark.parametrize(
    ("input_dict", "expected_result"),
    (
            (
                    {
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
            (
                    {
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


@pytest.mark.parametrize(
    ("input_dict", "scoring", "expected_result"),
    (
            (
                    {
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

            (
                    {
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
             {"roc_auc_score": 0.6666666666666667, "accuracy_score": 0.52}),
            (LogisticRegression(C=10.0, tol=1, random_state=42), {"roc_auc_score": 0.6466666666666666, "accuracy_score": 0.48}),
            (DecisionTreeClassifier(max_depth=100, criterion="gini", random_state=42),
             {"roc_auc_score": 0.5333333333333333, "accuracy_score": 0.6}),
            (SVC(C=1.0, tol=0.1, random_state=1), {"roc_auc_score": 0.5999999999999999, "accuracy_score": 0.6})
    )
)
def test_model_finder_score_model(model_finder_classification, model, expected_results):
    """Testing if score_model() function properly scores different Models."""
    X_train, y_train = model_finder_classification.X_train, model_finder_classification.y_train
    model.fit(X_train, y_train)
    model_finder_classification.scoring_functions = [roc_auc_score, accuracy_score]
    actual_results = model_finder_classification._score_model(model, roc_auc_score)

    assert actual_results == expected_results


def test_model_finder_test_target_proportion(model_finder_classification_fitted):
    """Testing if proportion of 1s in target (y_test) is calculated correctly (in classification)."""
    assert model_finder_classification_fitted.test_target_proportion() == 0.6

