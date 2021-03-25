import random
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, train_test_split, HalvingGridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.exceptions import NotFittedError

import time
import datetime
import warnings

from .models import classifiers


class ModelFinder:

    _classification = "classification"
    _regression = "regression"
    _quick_search_limit = 3
    _available_scoring_classification = [accuracy_score, roc_auc_score]

    __modes = ["quick", "detailed"]
    __target_categories = ["categorical", "numerical"]

    def __init__(self, target_category):

        if target_category.lower() in self.__target_categories:
            if target_category.lower() == "categorical":
                self.problem = self._classification
            elif target_category.lower() == "numerical":
                self.problem = self._regression
        else:
            raise ValueError("Expected one of the categories: {categories}; got {category}".format(
                categories=", ".join(self.__target_categories), category=target_category
            ))

        self.chosen_model = None
        self.chosen_model_params = None

        self.all_models = None

        self._quicksearch_results = None
        self._gridsearch_results = None

    def search_and_fit(self, X, y, models=None, scoring=None, mode="quick", random_state=None):
        """models can be either:
            - list of initialized models, to which we fit the data
            - dict of Model (class): param_grid of a given model to do the GridSearch
            - None, which will lead to the usage of default list of models, based on a provided "mode"

        mode can be either:
            - "quick": search is initially done on all models but with no parameter tuning after which top
                ._quick_search_limit number is chosen from the best models and GridSearched
            - "detailed": GridSearch on all default models and default params
        """
        # TODO: separate classification and regression
        # TODO: add TargetRegressor in case of regression
        # TODO: decide where random state is needed

        if mode not in self.__modes:
            raise ValueError("expected one of the modes: {modes}; got {mode}".format(
                modes=", ".join(self.__modes), mode=mode
            ))

        if self.problem == self._classification:
            default_models = classifiers

        # TODO: implement Stratified split in case of imbalance
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25)

        if isinstance(models, dict):
            clfs = self._gridsearch(X_train, y_train, models, scoring)
        elif models is None:
            if mode == "quick":
                # TODO: make distinction between classification and regression
                chosen_models = self._quicksearch(X_train, y_train, default_models.keys(), scoring)
                param_grid = {clf: default_models[clf] for clf in chosen_models}
                clfs = self._gridsearch(X_train, y_train, param_grid, scoring)
                pass
            elif mode == "detailed":
                clfs = self._gridsearch(X_train, y_train, default_models, scoring)
                pass
        else:
            # clfs = fit models X_train, y_train
            pass

        # Predict and score all clfs
        # Choose the best model

    def _gridsearch(self, X, y, models_param_grid, scoring):
        # TODO: decide whether random state is needed

        result_df = None
        best_of_their_class = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25)

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
                clf.fit(X_train, y_train)
                stop_time = datetime.datetime.now()

                results = pd.DataFrame(clf.cv_results_)
                results["model"] = model.__name__
                results["start"] = start_time
                results["stop_time"] = stop_time
                result_df = pd.concat([result_df, results], axis=0)

                best_params = clf.best_params_
                clf = model(**best_params)
                best_of_their_class.append(clf)

        self._gridsearch_results = result_df
        return best_of_their_class

    def _quicksearch(self, X, y, models, scoring):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        results = []
        for model in models:
            clf = model().fit(X_train, y_train)
            score = scoring(y_test, clf.predict(X_test))
            results.append((model, score))

        results.sort(key=lambda x: x[1], reverse=True)
        self._quicksearch_results = results
        return results


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