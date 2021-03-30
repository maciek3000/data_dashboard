import random
import pandas as pd
import time
import warnings
import copy

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, train_test_split, HalvingGridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.exceptions import NotFittedError

from .models import classifiers, regressors


def name(obj):
    try:
        obj_name = obj.__name__
    except AttributeError:
        obj_name = type(obj).__name__
    return obj_name


class ModelNotSetError(ValueError):
    pass


class ModelFinder:

    _classification = "classification"
    _regression = "regression"
    _quick_search_limit = 3
    _scoring_classification = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score]
    _scoring_regression = [mean_squared_error, mean_absolute_error, explained_variance_score, r2_score]

    _model_name = "model"

    _mode_quick = "quick"
    _mode_detailed = "detailed"
    _modes = [_mode_quick, _mode_detailed]
    _target_categorical = "categorical"
    _target_numerical = "numerical"
    _target_categories = [_target_categorical, _target_numerical]

    def __init__(self, X, y, X_train, X_test, y_train, y_test, target_type, random_state=None):

        if target_type in self._target_categories:
            self._set_problem(target_type)
        else:
            raise ValueError("Expected one of the categories: {categories}; got {category}".format(
                categories=", ".join(self._target_categories), category=target_type
            ))

        self.random_state = random_state

        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self._chosen_model = None
        self._chosen_model_params = None
        self._chosen_model_scores = None
        self._quicksearch_results = None
        self._gridsearch_results = None
        self._search_results = None

        self._dummy_model, self._dummy_model_scores = self._create_dummy_model()

    def search_and_fit(self, models=None, scoring=None, mode=_mode_quick):
        # TODO: decide where random state is needed
        model = self.search(models, scoring, mode)
        self.set_model(model)
        self.fit()
        return self._chosen_model

    def set_and_fit(self, model):
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
                iter(models)

                # in case of Str
                if isinstance(models, str):
                    raise TypeError

                initiated_models = models
            except TypeError:
                raise ValueError("models should be Dict, List-like or None, got {models}".format(models=models))

        scored_models, search_results = self._assess_models(initiated_models, scoring)
        self._update_search_results(search_results)

        # assuming that scoring is: higher == better
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models[0][0]

    def set_model(self, model):
        self._chosen_model = model
        self._chosen_model_params = model.get_params()
        copy_for_scoring = copy.deepcopy(model).fit(self.X_train, self.y_train)
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

    def _set_problem(self, problem_type):

        if problem_type == self._target_categorical:
            self.problem = self._classification
            self.scoring_functions = self._scoring_classification
            self.default_models = classifiers
            self.default_scoring = roc_auc_score

        elif problem_type == self._target_numerical:
            self.problem = self._regression
            self.scoring_functions = self._scoring_regression
            self.default_models = regressors
            self.default_scoring = mean_squared_error

        return None

    def _search_for_models(self, models, mode, scoring):
        if isinstance(models, dict):
            gridsearch_models = self._gridsearch(models, scoring)

        elif models is None:
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

        initiated_models = [model(**params) for model, params in gridsearch_models]

        return initiated_models

    def _gridsearch(self, models_param_grid, scoring):
        chosen_models, all_results = self._perform_gridsearch(models_param_grid, scoring)
        self._update_gridsearch_results(all_results)
        return chosen_models

    def _perform_gridsearch(self, models_param_grid, scoring, cv=5):

        all_results = {}
        best_of_their_class = []

        for model, params in models_param_grid.items():

            # not every model requires random_state, e.g. KNeighborsClassifier
            if "random_state" in model().get_params().keys():
                params["random_state"] = [self.random_state]

            # # TODO: make it work
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")

            if self.problem == self._regression:
                # TODO: introduce function to properly make scorer (e.g. r2_score should have greater_is_better=True)
                scorer = make_scorer(scoring, greater_is_better=False)
            else:
                scorer = make_scorer(scoring)

            # GridSearch will fail with NotFittedError("All estimators failed to fit") when argument provided
            # in the param grid is incorrect for a given model (even one combination will trigger it).
            clf = HalvingGridSearchCV(
                model(),
                params,
                scoring=scorer,
                cv=cv,
                error_score=0,  # to ignore errors that might happen,
                random_state=self.random_state
            )
            try:
                clf.fit(self.X_train, self.y_train)
            except NotFittedError:
                # TODO: issue warning or print message that there might be incorrect arguments in param grid
                raise

            all_results[model] = clf.cv_results_
            best_of_their_class.append((model, clf.best_params_))

        return best_of_their_class, all_results,

    def _update_gridsearch_results(self, cv_results):
        df = None
        for model in cv_results.keys():
            single_results = pd.DataFrame(cv_results[model])
            single_results[self._model_name] = name(model)
            df = pd.concat([df, single_results], axis=0)

        df = df.reset_index().drop(["index"], axis=1)
        self._gridsearch_results = df

    def _quicksearch(self, models, scoring):
        scored_models, all_results = self._perform_quicksearch(models, scoring)
        self._update_quicksearch_results(all_results)

        # assuming that scoring is: higher == better
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return [model[0] for model in scored_models][:self._quick_search_limit]

    def _perform_quicksearch(self, models, scoring):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train)

        all_results = {}
        scored_models = []
        for model in models:
            start_time = time.time()
            clf = model().fit(X_train, y_train)
            stop_time = time.time()
            score = scoring(y_test, clf.predict(X_test))
            params = clf.get_params()

            all_results[name(model)] = {"fit_time": stop_time-start_time, name(scoring): score, "params": params}
            scored_models.append((model, score))

        return scored_models, all_results

    def _update_quicksearch_results(self, results):
        df = pd.DataFrame(results).T
        self._quicksearch_results = df

    def _assess_models(self, models, chosen_scoring):
        all_results = {}
        scored_models = []

        for model in models:
            model_copy = copy.deepcopy(model)
            start_time = time.time()
            model_copy.fit(self.X_train, self.y_train)
            stop_time = time.time()

            model_results = {"fit_time": stop_time-start_time, "params": model.get_params()}
            score_results = self._score_model(model_copy, chosen_scoring)
            model_results.update(score_results)

            all_results[name(model)] = model_results
            scored_models.append((model, score_results[chosen_scoring]))

        return scored_models, all_results

    def _update_search_results(self, results):
        df = pd.DataFrame(results).T
        self._search_results = df

    def _score_model(self, model, chosen_scoring):

        if chosen_scoring in self.scoring_functions:
            scorings = self.scoring_functions
        else:
            scorings = [chosen_scoring] + self.scoring_functions

        scoring_results = {}
        for scoring in scorings:
            score = scoring(self.y_test, model.predict(self.X_test))
            scoring_results[scoring] = score

        return scoring_results

    def _create_dummy_model(self):
        if self.problem == self._classification:
            model = DummyClassifier(strategy="stratified", random_state=self.random_state)
        elif self.problem == self._regression:
            model = DummyRegressor(strategy="median")
        else:
            raise Exception("?")

        model.fit(self.X_train, self.y_train)
        results = self._score_model(model, self.default_scoring)
        model.fit(self.X, self.y)
        return model, results
