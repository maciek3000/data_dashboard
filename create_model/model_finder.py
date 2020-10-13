import random
import numpy as np

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split


class ModelFinder:

    def __init__(self, models_grid, X, y, scoring, gridsearch_kwargs=None, logs=True, random_state=None):
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

        self.models_grid = models_grid
        self.models = models_grid.keys()

        self.X = X
        self.y = y
        self.scoring = scoring

        self.gridsearch_kwargs = gridsearch_kwargs

        self.logs = logs

        if random_state:
            self.random_state = random_state
        else:
            self.random_state = random.randint(0, 10000)

    def find_best_model(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=self.random_state)

        models_scores = []

        # TODO: logging
        for model in self.models:
            # Continues with another params and models if some params are incompatible
            # TODO: decide if needed
            try:
                clf = GridSearchCV(
                    model(random_state=self.random_state),
                    self.models_grid[model],
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
