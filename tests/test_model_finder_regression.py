import pytest
import numpy as np
import pandas as pd
import copy
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import NotFittedError
from sklearn.dummy import DummyRegressor

from ml_dashboard.model_finder import ModelsNotSearchedError
from ml_dashboard.model_finder import WrappedModelRegression


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


def test_model_finder_set_model_regression(model_finder_regression, seed):
    """Testing if set_model() function correctly sets chosen Model and corresponding properties (regression).
    Additionally checks if the set Model wasn't fitted in the process."""
    model = DecisionTreeRegressor(max_depth=10, criterion="mae", random_state=seed)
    mf = model_finder_regression
    mf.scoring_functions = [mean_squared_error, r2_score]
    mf.set_model(model)

    assert mf._chosen_model.clf.regressor == model
    assert mf._chosen_model_params == model.get_params()
    assert mf._chosen_model_scores == {"mean_squared_error": 1323.0223273506956, "r2_score": -1.7265322117249737}
    assert type(mf._chosen_model) == WrappedModelRegression

    with pytest.raises(NotFittedError):
        mf.predict([1])


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
    ("mode", "expected_model", "expected_scores"),
    (
            ("quick", SVR(C=0.1, tol=1.0), {"mean_squared_error": 486.38607353926875, "r2_score": -0.002361993009452945}),
            ("detailed", SVR(C=0.1, tol=1.0), {"mean_squared_error": 486.38607353926875, "r2_score": -0.002361993009452945})
    )
)
def test_model_finder_search_and_fit_regression(model_finder_regression, mode, expected_model, expected_scores, seed):
    """Testing if search_and_fit() function correctly searches for and sets and fits chosen model (regression).
    Additionally checks if the model is correctly wrapped in TransformedTargetRegressor."""
    prediction_array = np.array([1.34, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(1, -1)
    model_finder_regression._quicksearch_limit = 1
    model_finder_regression.scoring_functions = [mean_squared_error, r2_score]
    actual_model = model_finder_regression.search_and_fit(models=None, scoring=mean_squared_error, mode=mode)
    expected_model.random_state = seed
    t_X = model_finder_regression.X
    t_y = model_finder_regression.y
    m = TransformedTargetRegressor(regressor=expected_model, transformer=QuantileTransformer(output_distribution="normal", random_state=seed))
    m.fit(t_X, t_y)
    expected_array = m.predict(prediction_array)

    assert str(actual_model) == str(expected_model)
    assert str(model_finder_regression._chosen_model) == str(expected_model)
    assert model_finder_regression._chosen_model_params == expected_model.get_params()
    assert model_finder_regression._chosen_model_scores == expected_scores
    assert type(actual_model) == WrappedModelRegression
    assert str(actual_model.clf.regressor) == str(expected_model)

    assert np.array_equal(model_finder_regression.predict(prediction_array), expected_array)


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
    (including models not present in default models collection)."""
    actual_model = model_finder_regression.search(models=models, scoring=mean_squared_error)
    assert str(actual_model) == str(expected_model)


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


def test_model_finder_perform_quicksearch_regression(model_finder_regression, chosen_regressors_grid, seed):
    """Testing if quicksearch works and returns correct Models and result dict (in regression)."""
    expected_models = [
        (DecisionTreeRegressor, 1801.9747017092634),
        (Ridge, 1530.5096036958037),
        (SVR, 1197.791404746514),
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
    ("limit",),
    (
            (1,),
            (2,),
    )
)
def test_model_finder_regression_search_results_dataframe(model_finder_regression_fitted, limit, seed):
    """Testing if search_results_dataframe is being correctly filtered out to a provided
    model_limit (in regression)"""
    models = ["SVR", "Ridge", "DecisionTreeRegressor"]
    dummy = ["DummyRegressor"]
    expected_index = models[:limit] + dummy
    expected_keys = {
        "fit_time", "params", "mean_squared_error", "r2_score", "mean_absolute_error", "explained_variance_score"
    }

    actual_results = model_finder_regression_fitted.search_results(limit)

    assert actual_results.index.tolist() == expected_index
    assert set(actual_results.columns) == expected_keys


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_regression_prediction_errors(model_finder_regression_fitted, limit, seed):
    """Testing if calculated prediction errors are correct (for regression)."""
    results = [
        SVR(**{"C": 0.1, "tol": 1.0}),
        Ridge(**{"alpha": 0.0001, "random_state": seed}),
        DecisionTreeRegressor(**{"max_depth": 10, "criterion": "mae", "random_state": seed}),
    ]
    expected_results = results[:limit]
    expected_len = 25

    actual_results = model_finder_regression_fitted.prediction_errors(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert str(actual_result[0]) == str(expected_result)
        y_score, y_pred = actual_result[1]
        assert y_score.shape[0] == expected_len
        assert y_pred.shape[0] == expected_len

        assert not np.array_equal(y_score, y_pred)


def test_model_finder_regression_prediction_errors_error(model_finder_regression):
    """Testing if prediction_errors raises an error when there are no search results available (regression)."""
    with pytest.raises(ModelsNotSearchedError) as excinfo:
        _ = model_finder_regression.prediction_errors(1)
    assert "Search Results is not available. " in str(excinfo.value)


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_regression_residuals(model_finder_regression_fitted, limit, seed):
    """Testing if calculated residuals are correct (for regression)."""
    results = [
        SVR(**{"C": 0.1, "tol": 1.0}),
        Ridge(**{"alpha": 0.0001, "random_state": seed}),
        DecisionTreeRegressor(**{"max_depth": 10, "criterion": "mae", "random_state": seed}),
    ]
    expected_results = results[:limit]
    expected_len = 25

    actual_results = model_finder_regression_fitted.residuals(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert str(actual_result[0]) == str(expected_result)
        y_score, y_pred = actual_result[1]
        assert y_score.shape[0] == expected_len
        assert y_pred.shape[0] == expected_len

        assert not np.array_equal(y_score, y_pred)


def test_model_finder_regression_residual_error(model_finder_regression):
    """Testing if prediction_errors raises an error when there are no search results available (regression)."""
    with pytest.raises(ModelsNotSearchedError) as excinfo:
        _ = model_finder_regression.residuals(1)
    assert "Search Results is not available. " in str(excinfo.value)


@pytest.mark.parametrize(
    ("limit",),
    (
            (1,),
            (2,),
            (3,),
    )
)
def test_model_finder_predict_X_test_regression(model_finder_regression_fitted, split_dataset_numerical, limit, seed):
    """Testing if predictions of X_test split from found models are correct (in regression)."""
    models = [
        SVR(**{"C": 0.1, "tol": 1.0}),
        Ridge(**{"alpha": 0.0001, "random_state": seed}),
        DecisionTreeRegressor(**{"max_depth": 10, "criterion": "mae", "random_state": seed}),
    ]
    results = []
    X_train, X_test, y_train, y_test = split_dataset_numerical
    transformer = QuantileTransformer(output_distribution="normal", random_state=seed)
    for model in models:
        new_model = TransformedTargetRegressor(regressor=model, transformer=transformer)
        new_model.fit(X_train, y_train)
        results.append((model, new_model.predict(X_test)))

    expected_results = results[:limit]

    actual_results = model_finder_regression_fitted.predictions_X_test(limit)

    for actual_result, expected_result in zip(actual_results, expected_results):
        assert str(actual_result[0]) == str(expected_result[0])
        assert np.array_equal(actual_result[1], expected_result[1])


@pytest.mark.parametrize(
    ("expected_model",),
    (
            (Ridge(),),
            (DecisionTreeRegressor(max_depth=5),),
            (SVR(C=1000),)
    )
)
def test_model_finder_wrap_model_regression(model_finder_regression, expected_model):
    """Testing if wrapping a chosen model with custom TransformedTargetRegressor works correctly (regression)."""
    actual_model = model_finder_regression._wrap_model(expected_model)

    assert str(actual_model) == str(expected_model)
    assert actual_model.get_params() == expected_model.get_params()
    assert type(actual_model) == WrappedModelRegression
    assert actual_model.clf.regressor == expected_model


@pytest.mark.parametrize(
    ("test_data",),
    (
            ({"A": [1, 2, 3], "B": [10, 20, 30]},),
            ({"aa": [1, 2], "B": [1, 2], "C": [3, 3], "last": [4, 100]},)
    )
)
def test_model_finder_wrap_results_dataframe_regression(model_finder_regression, test_data):
    """Testing if adding additional column to the results dataframe works correctly (in regression)."""
    df = pd.DataFrame(data=test_data)
    title = model_finder_regression._transformed_target_name
    params = str(model_finder_regression.target_transformer.get_params())
    actual_df = model_finder_regression._wrap_results_dataframe(df.copy())
    expected_df = df.copy()
    expected_df[title] = params

    assert actual_df.equals(expected_df)


@pytest.mark.parametrize(
    ("test_params",),
    (
            ({"A": 100, "B": "test", "C": [1, 2]},),
            ({"uno": 1, "dos": 2, "tres": "quatro"},)
    )
)
def test_model_finder_wrap_params_regression(model_finder_regression, test_params):
    """Testing if adding params from Target Transformer to params dict works correctly (in regression)."""
    expected_params = copy.deepcopy(test_params)
    title = model_finder_regression._transformed_target_name
    transformer = model_finder_regression.target_transformer
    new_params = {title + "__" + key: item for key, item in transformer.get_params().items()}
    new_params[title] = transformer.__class__.__name__

    expected_params.update(new_params)
    actual_params = model_finder_regression._wrap_params(copy.deepcopy(test_params))
    assert actual_params == expected_params


@pytest.mark.parametrize(
    ("model",),
    (
            (Ridge(),),
            (DecisionTreeRegressor(max_depth=5),),
            (SVR(C=1000),)
    )
)
def test_model_finder_calculate_model_score_multiclass_regular_scoring(model_finder_regression, split_dataset_numerical, model):
    """Testing if calculating model score works correctly in multiclass with scoring != roc_auc_score."""
    scoring = mean_squared_error
    X_train = split_dataset_numerical[0]
    X_test = split_dataset_numerical[1]
    y_train = split_dataset_numerical[2]
    y_test = split_dataset_numerical[3]

    model.fit(X_train, y_train)

    expected_result = scoring(y_test, model.predict(X_test))
    actual_result = model_finder_regression._calculate_model_score(model, X_test, y_test, scoring)

    assert actual_result == expected_result
