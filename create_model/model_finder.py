import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier, DummyRegressor

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, train_test_split, HalvingGridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.exceptions import NotFittedError

import time
import datetime
import warnings
import copy

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

        self.chosen_model = None
        self.chosen_model_params = None
        self._quicksearch_results = None
        self._gridsearch_results = None
        self._compare_results = None

        self.dummy_model = self._create_dummy_model()

    def find_and_fit(self, models=None, scoring=None, mode=__mode_quick):
        # TODO: decide where random state is needed
        model = self.find(models, scoring, mode)
        self.set_model(model)
        self.fit()
        return self.chosen_model

    def set_and_fit(self, model):
        self.set_model(model)
        self.fit()

    def find(self, models=None, scoring=roc_auc_score, mode=__mode_quick):
        """models can be either:
            - list of initialized models, to which we fit the data
            - dict of Model (class): param_grid of a given model to do the GridSearch
            - None, which will lead to the usage of default list of models, based on a provided "mode"

        mode can be either:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                ._quick_search_limit number is chosen from the best models and GridSearched
            - "detailed": GridSearch on all default models and default params
        """
        # TODO: add separate fit_model function
        # TODO: add TargetRegressor in case of regression

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


        # TODO: I dont like this part
        search_results = self._compare_classifiers(self.X_train, self.X_test, self.y_train, self.y_test, initiated_models, scoring)
        chosen_model, chosen_model_params = self._choose_best_model(search_results, scoring)

        return chosen_model

    def set_model(self, model):
        self.chosen_model = model
        self.chosen_model_params = model.get_params()

    def fit(self):
        self.chosen_model.fit(self.X, self.y)

    def predict(self, X):
        return self.chosen_model.predict(X)

    def _set_problem(self, problem_type):
        if problem_type == self.__target_categorical:
            self.problem = self._classification
            self.scoring_functions = self._scoring_classification
            self.default_models = classifiers
        elif problem_type == self.__target_numerical:
            self.problem = self._regression
            self.scoring_functions = self._scoring_regression
            self.default_models = regressors
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

    def _compare_classifiers(self, X_train, X_test, y_train, y_test, models, chosen_scoring):
        results = []

        if chosen_scoring in self.scoring_functions:
            scorings = self.scoring_functions
        else:
            scorings = [chosen_scoring] + self.scoring_functions

        for model in models:
            start_time = time.time()
            model.fit(X_train, y_train)
            stop_time = time.time()
            # TODO: PassiveAgressiveClassifier has no name
            try:
                name = model.__name__
            except AttributeError:
                name = str(model)
            model_results = {"model": model, "fit_time": stop_time-start_time, "params": model.get_params()}
            for scoring in scorings:
                score = scoring(y_test, model.predict(X_test))
                model_results[scoring.__name__] = score

            results.append(model_results)

        self._compare_results = pd.DataFrame(results)
        return self._compare_results

    def _choose_best_model(self, results, chosen_scoring):
        scoring_name = chosen_scoring.__name__

        results_df = results.sort_values(by=[scoring_name], axis=0, ascending=False)
        model = results_df["model"].loc[0]
        params = results_df["params"].loc[0]
        return model, params

    def _create_dummy_model(self):
        if self.problem == self._classification:
            model = DummyClassifier(strategy="stratified")
        elif self.problem == self._regression:
            model = DummyRegressor(strategy="median")
        else:
            raise Exception("?")
        return model

