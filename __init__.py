from create_model.model_finder import ModelFinder

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



if __name__ == "__main__":
    d = load_digits(as_frame=True)

    X = d["data"]
    y = d["target"]

    model_rfc = RandomForestClassifier
    param_grid_rfc = {
        "n_estimators": [10, 100],
        "max_depth": [5, 10, 50],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2]
    }

    model_lr = LogisticRegression
    param_grid_lr = {
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["liblinear", "lbfgs", "saga"],
        "C": [0.1, 0.5, 1.0]
    }

    model_svc = SVC
    param_grid_svc = {
        "C": [0.1, 0.5, 1.0],
        "kernel": ["linear", "poly", "rbf"]
    }


    models_grid = {
        model_rfc: param_grid_rfc,
        model_lr: param_grid_lr,
        model_svc: param_grid_svc,
    }

    random_state = 2862

    finder = ModelFinder(models_grid, X, y, accuracy_score, random_state=random_state)
    model, score, params = finder.find_best_model()

    print("Model: {}\nScore: {}\nParams: {}".format(model.__name__, score, params))