"""Default models and their parameters to be used during search method of ModelFinder.

Note:
    Models and their parameters were chosen arbitrarily to achieve a good balance between performance and search
    results. By no means should they be treated as an exhaustive Models list, more as a quick reference for some
    of the Models available in sklearn.

Attributes:
    alpha (numpy.ndarray): array of alpha parameter values
    C (numpy.ndarray): array of C parameter values
    classifiers (dict): 'Model class': param_grid dict pairs of Models to be used in Classification/Multiclass
    regressors (dict): 'Model class': param_grid dict pairs of Models to be used in Regression
"""
import numpy as np
# Classification/Multiclass
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Regression
from sklearn.linear_model import HuberRegressor, Lasso, SGDRegressor, ElasticNet
from sklearn.linear_model import Ridge, PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


alpha = np.logspace(-4, 5, 10)
# tol = np.logspace(-5, -1, 5)
C = np.logspace(-4, 5, 10)

classifiers = {

    # SVM with Stochastic Gradient Descent Learning
    SGDClassifier: {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        "alpha": alpha,
        # "penalty": ["l1", "l2", "elasticnet"],  # l1 and elastic net bring sparsity to the model
    },

    # large scale learning, similar to Perceptron
    PassiveAggressiveClassifier: {
        "C": C,
        # "tol": tol
    },

    SVC: {
        "C": C,
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        # "gamma": ["scale", "auto"]
    },

    RidgeClassifier: {
        "alpha": alpha,
        "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
        # "tol": tol,
    },

    Perceptron: {
        "alpha": alpha,
        "penalty": ["l1", "l2", "elasticnet"],
        # "tol": tol
    },

    LogisticRegression: {
        "C": C,
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["l1", "l2", "elasticnet"],
        # "tol": tol,
    },

    # fitting additional classifiers on the dataset with adjusted weights of incorrectly classified instances
    # default estimator is DecisionTreeClassifier
    AdaBoostClassifier: {
        "n_estimators": [10, 50, 100],
        "learning_rate": [0.01, 0.1, 1.0, 10.0, 100.0]
    },

    RandomForestClassifier: {
        "n_estimators": [10, 100, 200],
        "max_depth": [None, 10, 100],
        "min_samples_split": [2, 10, 0.01, 0.1],
        # "min_samples_leaf": [1, 2, 10, 0.01],
        # "max_features": ["sqrt", "log2"],
        # "min_impurity_decrease": [0, 1e-7, 1e-3]
        # "criterion": ["gini", "entropy"],
    },

    # fits on random subset of original data and aggregates their predictions
    # default estimator is DecisionTreeClassifier
    BaggingClassifier: {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
        # "bootstrap": [True, False]
    },

    # Random subset of features is used, but the thresholds for splitting on those features are random
    # decreases variance at the cost of increasing bias
    ExtraTreesClassifier: {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 10, 100],
        "min_samples_split": [2, 10, 0.01, 0.1],
        # "criterion": ["gini", "entropy"],
        # "min_samples_leaf": [1, 2, 10, 0.01],
        # "max_features": ["sqrt", "log2"]
    },

    KNeighborsClassifier: {
        "n_neighbors": [1, 2, 5, 10],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },

    XGBClassifier: {
        "max_depth": [1, 6, 10, 20],
        "eval_metric": ["logloss"],
        # "eta": np.logspace(-7, -3, 5),
        # "gamma": [0, 10, 100],
    },

    LGBMClassifier: {
        "max_depth": [3, 5, 7, -1],
        "min_child_samples": [1, 10, 20, 50]
    },

    MLPClassifier: {
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": alpha,
    }
}


regressors = {
    HuberRegressor: {
        "epsilon": np.linspace(1.1, 2.0, 9),
        "alpha": alpha
    },

    Lasso: {
        "alpha": np.logspace(1, 6, 6)
    },

    SGDRegressor: {
        "penalty": ["l1", "l2", "elasticnet"],
        "alpha": alpha,
        # "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
    },

    ElasticNet: {
        "alpha": np.logspace(1, 6, 6),
        "l1_ratio": np.linspace(0, 1, 4)
    },

    Ridge: {
        "alpha": alpha,
        # "tol": tol,
    },

    PassiveAggressiveRegressor: {
        "C": C,
    },

    SVR: {
        "C": C,
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    },

    RandomForestRegressor: {
        "n_estimators": [10, 100, 200],
        "max_depth": [None, 10, 100],
        "min_samples_split": [2, 10, 0.01, 0.1],
        # "n_estimators": [10, 100, 200],
        # "max_depth": [None, 10, 50],
        # "min_samples_split": [2, 0.001, 0.01, 0.1]
    },

    ExtraTreesRegressor: {
        "n_estimators": [10, 100, 200],
        "max_depth": [None, 10, 50],
        "min_samples_split": [2, 0.001, 0.01, 0.1]
    },

    AdaBoostRegressor: {
        "n_estimators": [10, 50, 100],
        "learning_rate": [0.01, 0.1, 1.0, 10.0, 100.0]
    },

    BaggingRegressor: {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
    },

    XGBRegressor: {
        "max_depth": [1, 6, 10, 20],
        "eval_metric": ["logloss"],
    },

    KNeighborsRegressor: {
        "n_neighbors": [1, 2, 5, 10],
        "weights": ["uniform", "distance"],
    },

    MLPRegressor: {
        "solver": ["lbfgs", "sgd", "adam"],
        "alpha": alpha
    }
}
