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


class ModelFinder:

    _classification = "classification"
    _regression = "regression"
    _quick_search_limit = 3
    _scoring_classification = [accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score]
    _scoring_regression = []

    __modes = ["quick", "detailed"]
    __target_categories = ["categorical", "numerical"]

    def __init__(self, X, y, target_type):

        self.X = X
        self.y = y

        if target_type in self.__target_categories:
            if target_type == "categorical":
                self.problem = self._classification
                self.scoring_functions = self._scoring_classification
                self.default_models = classifiers
            elif target_type == "numerical":
                self.problem = self._regression
                self.scoring_functions = self._scoring_regression
                self.default_models = regressors
        else:
            raise ValueError("Expected one of the categories: {categories}; got {category}".format(
                categories=", ".join(self.__target_categories), category=target_type
            ))

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


    def find_and_fit(self, models=None, scoring=roc_auc_score, mode="quick", random_state=None):
        # TODO: decide where random state is needed
        model = self.find(scoring, models, mode, random_state)
        self.set_model(model)
        self.fit()
        return self.chosen_model

    def set_and_fit(self, model):
        self.set_model(model)
        self.fit()

    def find(self, scoring, models=None, mode="quick", random_state=None):
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

        if isinstance(models, dict):
            initiated_models = self._gridsearch(self.X_train, self.y_train, models, scoring)

        elif models is None:

            if mode == "quick":
                chosen_models = self._quicksearch(self.X_train, self.y_train, self.default_models.keys(), scoring)
                param_grid = {clf: self.default_models[clf] for clf in chosen_models}
                initiated_models = self._gridsearch(self.X_train, self.y_train, param_grid, scoring)
            elif mode == "detailed":
                initiated_models = self._gridsearch(self.X_train, self.y_train, self.default_models, scoring)
            else:
                # this branch shouldn't be possible without object properties manipulation
                raise Exception("?")

        else:
            try:
                iter(models)
                initiated_models = models
            except TypeError:
                raise ValueError("models should be Dict, List-like or None, got {models}".format(models=models))


        # TODO: I dont like this part
        search_results = self._compare_classifiers(self.X_train, self.X_test, self.y_train, self.y_test, initiated_models, scoring)
        chosen_model, chosen_model_params = self._choose_best_model(search_results, scoring)

        return chosen_model(**chosen_model_params)

    def set_model(self, model):
        self.chosen_model = model
        self.chosen_model_params = model.get_params()

    def fit(self):
        self.chosen_model.fit(self.X, self.y)

    def predict(self, X):
        return self.chosen_model.predict(X)

    def _gridsearch(self, X, y, models_param_grid, scoring):
        # TODO: decide whether random state is needed

        result_df = None
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

                start_time = datetime.datetime.now()
                clf.fit(X, y)
                stop_time = datetime.datetime.now()

                results = pd.DataFrame(clf.cv_results_)
                results["model"] = model.__name__
                results["fit_time"] = stop_time - start_time
                result_df = pd.concat([result_df, results], axis=0)

                best_params = clf.best_params_
                clf = model(**best_params)
                best_of_their_class.append(clf)

        self._gridsearch_results = result_df
        return best_of_their_class

    def _quicksearch(self, X, y, models, scoring):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        results = []
        best_of_their_class = []
        for model in models:
            start_time = datetime.datetime.now()
            clf = model().fit(X_train, y_train)
            stop_time = datetime.datetime.now()
            score = scoring(y_test, clf.predict(X_test))
            params = clf.get_params()
            # TODO: PassiveAgressiveClassifier has no name
            try:
                name = model.__name__
            except AttributeError:
                name = "noname"
            results.append({"model": name, "fit_time": stop_time-start_time, "score": score, "params": params})
            # TODO: quicksearch shouldnt return initialized models
            best_of_their_class.append((model, score))

        results = pd.DataFrame(results)
        self._quicksearch_results = results
        # assuming that scoring is: higher == better
        best_of_their_class.sort(key=lambda x: x[1], reverse=True)

        return [model[0] for model in best_of_their_class][:self._quick_search_limit]

    def _compare_classifiers(self, X_train, X_test, y_train, y_test, models, chosen_scoring):
        results = []

        if chosen_scoring in self.scoring_functions:
            scorings = self.scoring_functions
        else:
            scorings = [chosen_scoring] + self.scoring_functions

        for model in models:
            start_time = datetime.datetime.now()
            model.fit(X_train, y_train)
            stop_time = datetime.datetime.now()
            # TODO: PassiveAgressiveClassifier has no name
            try:
                name = model.__name__
            except AttributeError:
                name = "noname"
            model_results = {"model": name, "fit_time": stop_time-start_time, "params": model.get_params()}
            for scoring in scorings:
                score = scoring(y_test, model.predict(X_test))
                model_results[scoring.__name__] = score

            results.append(model_results)

        self._compare_results = pd.DataFrame(results)
        return self._compare_results

    def _choose_best_model(self, results, chosen_scoring):
        scoring_name = chosen_scoring.__name__

        results_df = results.sort_values(by=[scoring_name], axis=1, ascending=False)
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

    # def _choose_model(self):
    #     # checking score on X_test and y_test
    #     # looped to account for different random_states and averaging the result
    #
    #     # there is also a possibility to use cross_val_score
    #     # https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    #
    #     model_score = []
    #     for i in range(5):
    #         m = model(**best_params)
    #         m.fit(X_train, y_train)
    #         score = self.scoring(y_test, m.predict(X_test))
    #         model_score.append(score)
    #
    #     models_scores.append((model, np.mean(model_score), best_params))
    #
    #     best_model = max(models_scores, key=lambda x: x[1])
    #
    #     print("Found the model - time elapsed {time}".format(time=time.time() - start_time))
    #
    #     return best_model

    # def find_best_model(self):
    #
    #     # if self.models_grid is None:
    #     #     models_grid = self.__get_default_grid()
    #     # else:
    #     #     models_grid = self.models_grid
    #
    #     clfs = classifiers
    #
    #     models_grid = {}
    #
    #     for clf in clfs:
    #         _ = classifiers[clf]
    #         model = _[0]
    #         params = _[1]
    #         models_grid[model] = params
    #
    #     X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)
    #
    #     models = models_grid.keys()
    #     models_scores = []
    #
    #     # TODO: logging
    #     start_time = time.time()
    #     print("Starting..")
    #     for model in models:
    #         # Continues with another params and models if some params are incompatible
    #         print(model.__name__ + " - start time {dt}".format(dt=datetime.datetime.now()))
    #         with warnings.catch_warnings():
    #             warnings.filterwarnings("ignore")
    #             try:
    #                 # TODO: decide if needed
    #                 clf = HalvingGridSearchCV(
    #                     model(),
    #                     models_grid[model],
    #                     scoring=make_scorer(self.scoring),
    #                     cv=5
    #                 )
    #
    #                 clf.fit(X_train, y_train)
    #
    #                 best_params = clf.best_params_
    #
    #                 # checking score on X_test and y_test
    #                 # looped to account for different random_states and averaging the result
    #
    #                 # there is also a possibility to use cross_val_score
    #                 # https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
    #
    #                 model_score = []
    #                 for i in range(5):
    #                     m = model(**best_params)
    #                     m.fit(X_train, y_train)
    #                     score = self.scoring(y_test, m.predict(X_test))
    #                     model_score.append(score)
    #
    #                 models_scores.append((model, np.mean(model_score), best_params))
    #
    #             except NotFittedError:
    #                 continue
    #
    #     best_model = max(models_scores, key=lambda x: x[1])
    #
    #     print("Found the model - time elapsed {time}".format(time=time.time() - start_time))
    #
    #     return best_model

    # def __get_default_grid(self):
    #
    #     model_rfc = RandomForestClassifier
    #     param_grid_rfc = {
    #         "n_estimators": [10, 100],
    #         "max_depth": [5, 10, 50],
    #         "criterion": ["gini", "entropy"],
    #         "min_samples_split": [2, 5, 10],
    #         # "min_samples_leaf": [1, 2]
    #     }
    #
    #     model_gbc = GradientBoostingClassifier
    #     param_grid_gbc = {
    #         "learning_rate": [0.1, 0.5, 0.9],
    #         "n_estimators": [10, 100],
    #         "min_samples_split": [2, 5, 10]
    #     }
    #
    #     model_lr = LogisticRegression
    #     param_grid_lr = {
    #         "penalty": ["l1", "l2", "elasticnet"],
    #         "solver": ["liblinear", "lbfgs", "saga"],
    #         "C": [0.1, 0.5, 1.0]
    #     }
    #
    #     model_svc = SVC
    #     param_grid_svc = {
    #         "C": [0.1, 0.5, 1.0, 2.0, 5.0],
    #         "kernel": ["linear", "poly", "rbf"]
    #     }
    #
    #
    #     models_grid = {
    #         model_rfc: param_grid_rfc,
    #         model_lr: param_grid_lr,
    #         model_svc: param_grid_svc,
    #         model_gbc: param_grid_gbc
    #     }
    #
    #     return models_grid
    #
    # def quick_search(self):
    #     models = classifiers
    #
    #     X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)
    #
    #     results = []
    #     for name, model_tuple in models.items():
    #         model = model_tuple[0]
    #         clf = model().fit(X_train, y_train)
    #         score = roc_auc_score(y_test, clf.predict(X_test))
    #         results.append((name, score))
    #
    #     results.sort(key=lambda x: x[1], reverse=True)
    #
    #     return results