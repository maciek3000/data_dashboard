import random
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split


class ModelFinder:

    def __init__(self, X, y, scoring, models_grid=None, gridsearch_kwargs=None, logs=True, random_state=None):
        """
        ModelFinder object, designed to find the best model for provided X and y datasets.
        It uses GridSearchCV to go through different parameters for different models.
        After exhaustive gridsearch, best_chosen model is fitted and its score is calculated with a provided scoring
        function (average from 5 different fittings to try to negate randomness of models).
        At the end, the model which got the best score is returned.

        :param models_grid: should be a dictionary with Model Class - param_grid collection of parameters
        :param X: DataFrame WITHOUT classification target
        :param y: Classification target
        :param scoring:
        :param gridsearch_kwargs:
        :param logs:
        :param random_state:
        """

        self.X = X
        self.y = y
        self.scoring = scoring

        self.models_grid = models_grid
        self.gridsearch_kwargs = gridsearch_kwargs

        self.logs = logs

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = random.randint(0, 10000)

    def find_best_model(self):

        if self.models_grid is None:
            models_grid = self.__get_default_grid()
        else:
            models_grid = self.models_grid

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)

        models = models_grid.keys()
        models_scores = []

        # TODO: logging
        for model in models:
            # Continues with another params and models if some params are incompatible
            # TODO: decide if needed
            try:
                clf = GridSearchCV(
                    model(random_state=self.random_state),
                    models_grid[model],
                    scoring=make_scorer(self.scoring),
                    cv=5
                )

                clf.fit(X_train, y_train)

                best_params = clf.best_params_

                # checking score on X_test and y_test
                # looped to account for different random_states and averaging the result

                # there is also a possibility to use cross_val_score
                # https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

                model_score = []
                for i in range(5):
                    m = model(**best_params)
                    m.fit(X_train, y_train)
                    score = self.scoring(y_test, m.predict(X_test))
                    model_score.append(score)

                models_scores.append((model, np.mean(model_score), best_params))
            except:
                continue

        best_model = max(models_scores, key=lambda x: x[1])

        return best_model

    def __get_default_grid(self):

        model_rfc = RandomForestClassifier
        param_grid_rfc = {
            "n_estimators": [10, 100],
            "max_depth": [5, 10, 50],
            "criterion": ["gini", "entropy"],
            "min_samples_split": [2, 5, 10],
            # "min_samples_leaf": [1, 2]
        }

        model_gbc = GradientBoostingClassifier
        param_grid_gbc = {
            "learning_rate": [0.1, 0.5, 0.9],
            "n_estimators": [10, 100],
            "min_samples_split": [2, 5, 10]
        }

        model_lr = LogisticRegression
        param_grid_lr = {
            "penalty": ["l1", "l2", "elasticnet"],
            "solver": ["liblinear", "lbfgs", "saga"],
            "C": [0.1, 0.5, 1.0]
        }

        model_svc = SVC
        param_grid_svc = {
            "C": [0.1, 0.5, 1.0, 2.0, 5.0],
            "kernel": ["linear", "poly", "rbf"]
        }


        models_grid = {
            model_rfc: param_grid_rfc,
            # model_lr: param_grid_lr,
            # model_svc: param_grid_svc,
            # model_gbc: param_grid_gbc
        }

        return models_grid