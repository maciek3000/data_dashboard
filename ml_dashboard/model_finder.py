import pandas as pd
import numpy as np
import time
import copy
import warnings
from collections import defaultdict

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.preprocessing import QuantileTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, det_curve
from sklearn.exceptions import NotFittedError

from .models import classifiers, regressors
from .functions import reverse_sorting_order, obj_name


class ModelNotSetError(ValueError):
    pass


class ModelsNotSearchedError(ValueError):
    pass


class WrappedModelRegression:

    def __init__(self, regressor, transformer):
        self.clf = TransformedTargetRegressor(regressor=regressor, transformer=transformer)
        self.__name__ = self.clf.regressor.__class__.__name__
        self.__class__ = self.clf.regressor.__class__

    def fit(self, *args, **kwargs):
        self.clf.fit(*args, **kwargs)
        return self

    def predict(self, *args, **kwargs):
        return self.clf.predict(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        return self.clf.regressor.get_params(*args, **kwargs)

    def __str__(self):
        return self.clf.regressor.__str__()

    def __class__(self, *args, **kwargs):
        return self.clf.regressor.__class__(*args, **kwargs)


class ModelFinder:
    """Used to search for the best Models possible with a brute force approach of GridSearching.

        ModelFinder takes as initial arguments the data (X, y) and the split of the aforementioned data (train/test).
        Then it uses train/test split to assess any model, but fits chosen models on all X, y.

        If no model is set or search() function is called explicitly, then ModelFinder does a search within all
        predefined Models (for a given problem type: classification/regression) and based on a provided scoring assesses
        which of those Models is the best. Based on a provided mode (quick or detailed), search will be performed:
            - quick: initial search is done on default variables of different classes of Models (similar to LazyPredict)
            and a certain number of Models is then GridSearched
            - detailed: GridSearch is done on ALL predefined Models.

        Model that was the best (based on the scoring) is chosen internally, fitted and can be then used to predict
        any data (e.g. for Kaggle competition).

        Additionally, results from the search are also available in object properties (as DataFrames) and can be
        accessed for additional insight.

        Please note that all X, y, train/test data needs to be already transformed.
    """

    _classification = "classification"
    _regression = "regression"
    _multiclass = "multiclass"
    _quicksearch_limit = 3
    _scoring_classification = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score]
    _scoring_regression = [mean_squared_error, mean_absolute_error, explained_variance_score, r2_score]
    _scoring_multiclass_parametrized = [
        (f1_score, {"average": "micro"}, "f1_score_micro"),
        (f1_score, {"average": "weighted"}, "f1_score_weighted"),
        (precision_score, {"average": "weighted"}, "precision_score_weighted"),
        (recall_score, {"average": "weighted"}, "recall_score_weighted"),
    ]
    _scoring_multiclass = [accuracy_score, balanced_accuracy_score]

    _model_name = "model"
    _fit_time_name = "fit_time"
    _params_name = "params"
    _transformed_target_name = "TransformedTargetRegressor__transformer"

    _mode_quick = "quick"
    _mode_detailed = "detailed"
    _modes = [_mode_quick, _mode_detailed]
    _target_categorical = "categorical"
    _target_numerical = "numerical"
    _target_categories = [_target_categorical, _target_numerical]
    _probas_functions = ["roc_auc_score"]

    def __init__(self, X, y, X_train, X_test, y_train, y_test, target_type, random_state=None):

        self.random_state = random_state

        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        if target_type in self._target_categories:
            self._set_problem(target_type)
        else:
            raise ValueError("Expected one of the categories: {categories}; got {category}".format(
                categories=", ".join(self._target_categories), category=target_type
            ))

        self._chosen_model = None
        self._chosen_model_params = None
        self._chosen_model_scores = None

        self._search_results = None
        self._search_results_dataframe = None

        self._quicksearch_results = None
        self._gridsearch_results = None

        self._dummy_model, self._dummy_model_scores = self._create_dummy_model()

    # ===== # Machine Learning Models functions

    def search_and_fit(self, models=None, scoring=None, mode=_mode_quick):
        model = self.search(models, scoring, mode)
        self.set_model(model)
        self.fit()
        return self._chosen_model

    def set_model_and_fit(self, model):
        self.set_model(model)
        self.fit()

    def search(self, models=None, scoring=None, mode=_mode_quick):
        """models can be either:
            - list of initialized models, to which we fit the data
            - dict of Model (class): param_grid of a given model to do the GridSearch - no need to include random_state
                here as it will get overwritten by random_state property
            - None, which will lead to the usage of default list of models, based on a provided "mode"

        mode can be either:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                ._quick_search_limit number is chosen from the best models and GridSearched
            - "detailed": GridSearch on all default models and default params
        """

        if scoring is None:
            scoring = self.default_scoring

        if mode not in self._modes:
            raise ValueError("Expected one of the modes: {modes}; got {mode}".format(
                modes=", ".join(self._modes), mode=mode
            ))

        if isinstance(models, dict) or models is None:
            initiated_models = self._search_for_models(models, mode, scoring)
        else:
            try:
                # in case of Str
                if isinstance(models, str):
                    raise TypeError
                iter(models)
                initiated_models = [self._wrap_model(single_model) for single_model in models]
            except TypeError:
                raise ValueError("models should be Dict, List-like or None, got {models}".format(models=models))

        scored_and_fitted_models, search_results = self._assess_models(initiated_models, scoring)
        self._search_results = scored_and_fitted_models
        self._search_results_dataframe = self._create_search_results_dataframe(search_results, scoring)

        sorting_order = reverse_sorting_order(obj_name(scoring))
        scored_and_fitted_models.sort(key=lambda x: x[1], reverse=sorting_order)

        best_model = scored_and_fitted_models[0][0]  # first item in the list, first item in the (model, score) tuple
        new_model = best_model.__class__(**best_model.get_params())

        return new_model

    def set_model(self, model):
        model = self._wrap_model(model)
        params = model.get_params()
        copy_for_scoring = model.__class__(**params).fit(self.X_train, self.y_train) # copy.copy(model).fit(self.X_train, self.y_train)  #
        self._chosen_model = model
        self._chosen_model_params = params
        self._chosen_model_scores = self._score_model(copy_for_scoring, self.default_scoring)

    def fit(self):
        if self._chosen_model is None:
            raise ModelNotSetError(
                "Model needs to be set before fitting. Call 'set_model' or 'search' for a model before trying to fit."
            )
        self._chosen_model.fit(self.X, self.y)

    def predict(self, X):
        if self._chosen_model is None:
            raise ModelNotSetError(
                "Model needs to be set and fitted before prediction. Call 'set_model' or 'search' for a model before."
            )
        return self._chosen_model.predict(X)

    def quicksearch_results(self):
        return self._quicksearch_results

    def gridsearch_results(self):
        return self._gridsearch_results

    # ===== # Visualization Data for View functions

    def search_results(self, model_limit):
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        # dummy is always included, regardless of model limit
        models = self._search_results_dataframe.iloc[:model_limit]
        dummy = self._dummy_model_results()

        df = pd.concat([models, dummy], axis=0).set_index(self._model_name)
        return df

    def dataframe_params_name(self):
        return self._params_name

    def test_target_proportion(self):
        # only meaningful in binary classification
        return self.y_test.sum() / self.y_test.shape[0]

    def roc_curves(self, model_limit):
        return self._plot_curves(roc_curve, model_limit)

    def precision_recall_curves(self, model_limit):
        return self._plot_curves(precision_recall_curve, model_limit)

    def det_curves(self, model_limit):
        return self._plot_curves(det_curve, model_limit)

    def confusion_matrices(self, model_limit):
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]

        _ = []
        for model in models:
            _.append((model, confusion_matrix(self.y_test, model.predict(self.X_test))))
        return _

    def prediction_errors(self, model_limit):
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]
        _ = []
        for model in models:
            _.append((model, (self.y_test, model.predict(self.X_test))))

        return _

    def residuals(self, model_limit):
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]
        _ = []
        for model in models:
            predictions = model.predict(self.X_test)
            _.append((model, (predictions, predictions - self.y_test)))

        return _

    def predictions_X_test(self, model_limit):
        if self._search_results_dataframe is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        models = [tp[0] for tp in self._search_results[:model_limit]]
        _ = []
        for model in models:
            _.append((model, model.predict(self.X_test)))

        return _

    # ===== # Internal functions

    def _set_problem(self, problem_type):

        if problem_type == self._target_categorical:

            if len(np.unique(self.y)) > 2:
                self.problem = self._multiclass
                self.scoring_functions = self._create_scoring_multiclass()
                self.default_models = classifiers
                self.default_scoring = self.scoring_functions[0]
            else:
                self.problem = self._classification
                self.scoring_functions = self._scoring_classification
                self.default_models = classifiers
                self.default_scoring = roc_auc_score

        elif problem_type == self._target_numerical:
            self.problem = self._regression
            self.scoring_functions = self._scoring_regression
            self.default_models = regressors
            self.default_scoring = mean_squared_error
            self.target_transformer = QuantileTransformer(output_distribution="normal", random_state=self.random_state)

        return None

    def _search_for_models(self, models, mode, scoring):
        if isinstance(models, dict):
            gridsearch_models = self._gridsearch(models, scoring)

        elif models is None:
            # filtering warnings from gridsearch/quicksearch (e.g. models not converging)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if mode == self._mode_quick:
                    chosen_models = self._quicksearch(self.default_models.keys(), scoring)
                    param_grid = {clf: self.default_models[clf] for clf in chosen_models}
                    gridsearch_models = self._gridsearch(param_grid, scoring)
                elif mode == self._mode_detailed:
                    gridsearch_models = self._gridsearch(self.default_models, scoring)
                else:
                    # this branch shouldn't be possible without explicit class/object properties manipulation
                    raise ValueError("models should be Dict or None, got {models}".format(models=models))
        else:
            raise ValueError("Expected one of the modes: {modes}; got {mode}".format(
                modes=", ".join(self._modes), mode=mode
            ))

        initiated_models = [self._wrap_model(model(**params)) for model, params in gridsearch_models]

        return initiated_models

    def _gridsearch(self, models_param_grid, scoring):
        chosen_models, all_results = self._perform_gridsearch(models_param_grid, scoring)
        self._gridsearch_results = self._wrap_results_dataframe(self._create_gridsearch_results_dataframe(all_results))
        return chosen_models

    def _perform_gridsearch(self, models_param_grid, scoring, cv=5):

        all_results = {}
        best_of_their_class = []

        for model, params in models_param_grid.items():

            # not every model requires random_state, e.g. KNeighborsClassifier
            if "random_state" in model().get_params().keys():
                params["random_state"] = [self.random_state]

            # https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
            sorting_order = reverse_sorting_order(obj_name(scoring))

            created_model = self._wrap_model(model())

            # GridSearch will fail with NotFittedError("All estimators failed to fit") when argument provided
            # in the param grid is incorrect for a given model (even one combination will trigger it).
            clf = HalvingGridSearchCV(
                created_model,
                params,
                scoring=make_scorer(scoring, greater_is_better=sorting_order),
                cv=cv,
                error_score=0,  # to ignore errors that might happen,
                random_state=self.random_state
            )
            try:
                clf.fit(self.X_train, self.y_train)
            except NotFittedError:
                # printing out warning for user as the NotFittedError might be misleading in this case
                params_str = ["{}: {}".format(key, item) for key, item in params.items()]
                warn_msg = "WARNING: {} might potentially have had incorrect params provided: {}".format(model, params_str)
                warnings.warn(warn_msg)
                raise

            all_results[model] = clf.cv_results_
            best_of_their_class.append((model, clf.best_params_))

        return best_of_their_class, all_results,

    def _create_gridsearch_results_dataframe(self, cv_results):
        df = None
        for model in cv_results.keys():
            single_results = pd.DataFrame(cv_results[model])
            single_results[self._model_name] = obj_name(model)
            df = pd.concat([df, single_results], axis=0)

        df = df.reset_index().drop(["index"], axis=1)
        return df

    def _quicksearch(self, models, scoring):
        scored_models, all_results = self._perform_quicksearch(models, scoring)
        self._quicksearch_results = self._wrap_results_dataframe(self._create_search_results_dataframe(all_results, scoring))

        sorting_order = reverse_sorting_order(obj_name(scoring))
        scored_models.sort(key=lambda x: x[1], reverse=sorting_order)
        return [model[0] for model in scored_models][:self._quicksearch_limit]

    def _perform_quicksearch(self, models, scoring):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, random_state=self.random_state)

        all_results = {}
        scored_models = []
        for model in models:
            if "random_state" in model().get_params().keys():
                clf = model(random_state=self.random_state)
            else:
                clf = model()

            clf = self._wrap_model(clf)

            start_time = time.time()
            clf.fit(X_train, y_train)
            stop_time = time.time()

            score = self._calculate_model_score(clf, X_test, y_test, scoring)
            #score = scoring(y_test, clf.predict(X_test))
            params = clf.get_params()

            all_results[model] = {
                self._fit_time_name: stop_time-start_time, obj_name(scoring): score, self._params_name: params
            }
            scored_models.append((model, score))

        return scored_models, all_results

    def _create_search_results_dataframe(self, results, chosen_scoring):
        data = defaultdict(list)
        for model, values in results.items():
            data[self._model_name].append(obj_name(model))
            for key, val in values.items():
                data[key].append(val)

        sorting = not reverse_sorting_order(obj_name(chosen_scoring))
        return pd.DataFrame(data).sort_values(by=[obj_name(chosen_scoring)], ascending=sorting)

    def _assess_models(self, initiated_models, chosen_scoring):
        all_results = {}
        fitted_and_scored = []

        for model in initiated_models:
            with warnings.catch_warnings():  # ignoring warnings, esp QuantileTransformer for Wrapper in Regression
                warnings.simplefilter("ignore")
                start_time = time.time()
                model.fit(self.X_train, self.y_train)
                stop_time = time.time()

            model_results = {self._fit_time_name: stop_time-start_time, self._params_name: self._wrap_params(model.get_params())}
            score_results = self._score_model(model, chosen_scoring)
            model_results.update(score_results)

            all_results[model] = model_results
            fitted_and_scored.append((model, score_results[obj_name(chosen_scoring)]))

        return fitted_and_scored, all_results

    def _score_model(self, fitted_model, chosen_scoring):

        scorings = self._get_scorings(chosen_scoring)

        scoring_results = {}
        for scoring in scorings:
            score = self._calculate_model_score(fitted_model, self.X_test, self.y_test, scoring)
            #score = scoring(self.y_test, fitted_model.predict(self.X_test))
            scoring_results[obj_name(scoring)] = score

        return scoring_results

    def _create_dummy_model(self):
        if self.problem == self._classification or self.problem == self._multiclass:
            model = DummyClassifier(strategy="stratified", random_state=self.random_state)
        elif self.problem == self._regression:
            model = DummyRegressor(strategy="median")
        else:
            raise Exception("?")

        model.fit(self.X_train, self.y_train)
        results = self._score_model(model, self.default_scoring)
        return model, results

    def _dummy_model_results(self):
        _ = {
            self._model_name: obj_name(self._dummy_model),
            self._fit_time_name: np.nan,
            self._params_name: str(self._dummy_model.get_params()),
            **self._dummy_model_scores
        }
        return pd.DataFrame(_, index=[9999])

    def _get_scorings(self, chosen_scoring):
        if chosen_scoring in self.scoring_functions:
            scorings = self.scoring_functions
        else:
            scorings = [chosen_scoring] + self.scoring_functions

        return scorings

    def _plot_curves(self, plot_func, model_limit):
        if self._search_results is None:
            raise ModelsNotSearchedError("Search Results is not available. Call 'search' to obtain comparison models.")

        curves = []

        models = [tp[0] for tp in self._search_results[:model_limit]]
        for model in models:
            try:
                y_score = model.predict_proba(self.X_test)
                # https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/_plot/base.py#L104
                # predictions for class 1 in a binary setting - second column
                y_score = y_score[:, 1]
            except AttributeError:
                y_score = model.decision_function(self.X_test)

            result = plot_func(self.y_test, y_score)
            curves.append((model, result))

        return curves

    def _create_scoring_multiclass(self):

        # (roc_auc_score, {"average": "weighted", "multi_class": "ovr"})
        # multiclass roc_auc_score requires probabilities for every class in comparison to simple predictions
        # required by other metrics, not included cause it breaks program flow

        scorings = []
        for scoring, params, fname in self._scoring_multiclass_parametrized:

            def closure():
                func_scoring = scoring
                func_params = params
                func_name = fname

                def make_scoring(y_true, y_score):
                    make_scoring.__name__ = func_name  # name change for results logging
                    return func_scoring(y_true, y_score, **func_params)
                return make_scoring

            scorings.append(closure())

        scorings += self._scoring_multiclass
        return scorings

    def _wrap_model(self, model):
        if self.problem == self._regression and type(model) != WrappedModelRegression:
            wrapped_model = WrappedModelRegression(regressor=model, transformer=self.target_transformer)
            return wrapped_model
        else:
            return model

    def _wrap_results_dataframe(self, df):
        if self.problem == self._regression:
            df[self._transformed_target_name] = str(self.target_transformer.get_params())

        return df

    def _wrap_params(self, params):
        if self.problem == self._regression:
            params[self._transformed_target_name] = obj_name(self.target_transformer)
            transformer_params = self.target_transformer.get_params()
            new_params = {self._transformed_target_name + "__" + key: item for key, item in transformer_params.items()}
            params.update(new_params)

        return params

    def _calculate_model_score(self, model, X, y_true, scoring_function):

        if any((func == obj_name(scoring_function) for func in self._probas_functions)):
            try:
                predictions = model.predict_proba(X)  # included for multiclass, but currently not implemented
                if self.problem == self._classification:
                    predictions = predictions[:, 1]
            except AttributeError:
                predictions = model.decision_function(X)
        else:
            predictions = model.predict(X)
        score = scoring_function(y_true, predictions)
        return score
