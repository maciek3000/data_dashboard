import random
import pandas as pd
import time
import warnings
import copy

from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, train_test_split, HalvingGridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score, mean_squared_error
from sklearn.exceptions import NotFittedError

from .models import classifiers, regressors


def name(obj):
    try:
        obj_name = obj.__name__
    except AttributeError:
        obj_name = str(obj)

    return obj_name


class ModelFinder:

    _classification = "classification"
    _regression = "regression"
    _quick_search_limit = 3
    _scoring_classification = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score]
    _scoring_regression = []

    _model_name = "model"

    __mode_quick = "quick"
    __mode_detailed = "detailed"
    __modes = [__mode_quick, __mode_detailed]
    __target_categorical = "categorical"
    __target_numerical = "numerical"
    __target_categories = [__target_categorical, __target_numerical]

    def __init__(self, X, y, target_type, random_state=None):

        if target_type in self.__target_categories:
            self._set_problem(target_type)
        else:
            raise ValueError("Expected one of the categories: {categories}; got {category}".format(
                categories=", ".join(self.__target_categories), category=target_type
            ))

        if random_state is None:
            self.random_state = random.random()
        else:
            self.random_state = random_state

        self.X = X
        self.y = y

        # TODO: implement Stratified split in case of imbalance
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25)
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

    def search_and_fit(self, models=None, scoring=None, mode=__mode_quick):
        # TODO: decide where random state is needed
        model = self.search(models, scoring, mode)
        self.set_model(model)
        self.fit()
        return self._chosen_model

    def set_and_fit(self, model):
        self.set_model(model)
        self.fit()

    def search(self, models=None, scoring=None, mode=__mode_quick):
        """models can be either:
            - list of initialized models, to which we fit the data
            - dict of Model (class): param_grid of a given model to do the GridSearch
            - None, which will lead to the usage of default list of models, based on a provided "mode"

        mode can be either:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                ._quick_search_limit number is chosen from the best models and GridSearched
            - "detailed": GridSearch on all default models and default params
        """

        if scoring is None:
            scoring = self.default_scoring

        if mode not in self.__modes:
            raise ValueError("expected one of the modes: {modes}; got {mode}".format(
                modes=", ".join(self.__modes), mode=mode
            ))

        if isinstance(models, dict) or models is None:
            initiated_models = self._search_for_models(models, mode, scoring)
        else:
            try:
                iter(models)
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
            raise Exception("Model needs to be set before fitting")
        self._chosen_model.fit(self.X, self.y)

    def predict(self, X):
        return self._chosen_model.predict(X)

    def _set_problem(self, problem_type):

        if problem_type == self.__target_categorical:
            self.problem = self._classification
            self.scoring_functions = self._scoring_classification
            self.default_models = classifiers
            self.default_scoring = roc_auc_score

        elif problem_type == self.__target_numerical:
            self.problem = self._regression
            self.scoring_functions = self._scoring_regression
            self.default_models = regressors
            self.default_scoring = mean_squared_error

        return None

    def _search_for_models(self, models, mode, scoring):
        if isinstance(models, dict):
            initiated_models = self._gridsearch(models, scoring)

        elif models is None:
            if mode == self.__mode_quick:
                chosen_models = self._quicksearch(self.default_models.keys(), scoring)
                param_grid = {clf: self.default_models[clf] for clf in chosen_models}
                initiated_models = self._gridsearch(param_grid, scoring)
            elif mode == self.__mode_detailed:
                initiated_models = self._gridsearch(self.default_models, scoring)
            else:
                # this branch shouldn't be possible without explicit class/object properties manipulation
                raise Exception("?")
        else:
            raise Exception("?")

        return initiated_models

    def _gridsearch(self, models_param_grid, scoring):
        chosen_models, all_results = self._perform_gridsearch(models_param_grid, scoring)
        self._update_gridsearch_results(all_results)
        return chosen_models

    def _perform_gridsearch(self, models_param_grid, scoring):

        all_results = {}
        best_of_their_class = []

        for model, params in models_param_grid.items():
            # TODO: make it work
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                clf = HalvingGridSearchCV(
                    model(),
                    params,
                    scoring=make_scorer(scoring),
                    cv=5,
                    error_score=0  # to ignore errors that might happen
                )

                clf.fit(self.X_train, self.y_train)

                all_results[model] = clf.cv_results_
                best_model = model(**clf.best_params_)
                best_of_their_class.append(best_model)

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

            # quicksearch doesnt initialize models cause they are being gridsearched later on
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
            model = DummyClassifier(strategy="stratified")
        elif self.problem == self._regression:
            model = DummyRegressor(strategy="median")
        else:
            raise Exception("?")

        model.fit(self.X_train, self.y_train)
        results = self._score_model(model, self.default_scoring)
        model.fit(self.X, self.y)
        return model, results
