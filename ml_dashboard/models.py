import numpy as np
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier, Perceptron, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier #, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_hist_gradient_boosting

alpha = np.logspace(-4, 5, 10)
tol = np.logspace(-5, -1, 5)
C = np.logspace(-2, 3, 6)

# TODO: uncomment params

classifiers = {

    # SVM with Stochastic Gradient Descent Learning
    SGDClassifier: {
        "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
        # "penalty": ["l1", "l2", "elasticnet"],  # l1 and elastic net bring sparsity to the model
        "alpha": alpha,
        #"tol": tol
    },

    # large scale learning, similar to Perceptron
    PassiveAggressiveClassifier: {
        "C": C,
        "tol": tol
    },

    SVC: {
        "C": C,
        "tol": tol,
        #"kernel": ["linear", "poly", "rbf", "sigmoid"],
        #"gamma": ["scale", "auto"]
    },

    RidgeClassifier: {
        "alpha": alpha,
        "tol": tol,
        # "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
    },


    Perceptron: {
        "alpha": alpha,
        "penalty": ["l1", "l2", "elasticnet"],
        # "tol": tol
    },


    LogisticRegression: {
        "C": C,
        "tol": tol,
        # "penalty": ["l1", "l2", "elasticnet"],
        # "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    },

    # fitting additional classifiers on the dataset with adjusted weights of incorrectly classified instances
    # default estimator is DecisionTreeClassifier
    AdaBoostClassifier: {
        "n_estimators": [10, 50, 100],
        "learning_rate": np.logspace(-1, 1, 3)
    },

    RandomForestClassifier: {
        "n_estimators": [10, 100, 200],
        "criterion": ["gini", "entropy"],
        # "max_depth": [None, 10, 100],
        #"min_samples_split": [2, 10, 0.01],
        #"min_samples_leaf": [1, 2, 10, 0.01],
        # "max_features": ["sqrt", "log2"],
        # "min_impurity_decrease": [0, 1e-7, 1e-3]
    },

    # fits on random subset of original data and aggregates their predictions
    # default estimator is DecisionTreeClassifier
    BaggingClassifier: {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
        #"max_features": [0.5, 0.7, 1.0],
        #"bootstrap": [True, False]
    },

    # Random subset of features is used, but the thresholds for splitting on those features are random
    # decreases variance at the cost of increasing bias

    ExtraTreesClassifier: {
        "n_estimators": [10, 50, 100],
        "criterion": ["gini", "entropy"],
        # "max_depth": [None, 10, 100],
        # "min_samples_split": [2, 10, 0.01],
        # "min_samples_leaf": [1, 2, 10, 0.01],
        # "max_features": ["sqrt", "log2"]
    },


    KNeighborsClassifier: {
        "n_neighbors": [1, 2, 5, 10],
        "weights": ["uniform", "distance"],
        # "p": [1, 2]
    },

    XGBClassifier: {
        #"eta": np.logspace(-7, -3, 5),
        #"gamma": [0, 10, 100],
        "max_depth": [1, 6, 10, 20],
        # "lambda": [1, 5, 10, 50],
        "alpha": alpha,
        "eval_metric": ["logloss"]
    },

    LGBMClassifier: {
        #"num_leaves": [1, 5, 10, 50, 100],
        "max_depth": [3, 5, 7],
        #"min_data_in_leaf": [10, 100, 1000]
    },


    #     GaussianNB: {
    #         "var_smoothing": {1e-9, 1e-5}
    #     },

    NearestCentroid: {
        "metric": ["manhattan", "euclidean"],
        "shrink_threshold": [None, 1e-5, 1e-2]
    },

    MLPClassifier: {
        "solver": ["lbfgs", "sgd"],
        "alpha": alpha
    }

    #     HistGradientBoostingClassifier: {
    #         "learning_rate": {1e-5, 1e-2, 1e-1, 0},
    #         "max_iter": {10, 100, 200},
    #         "max_leaf_nodes": {10, 31, 50, None},
    #         "max_depth": {4, 10, 30, None},
    #         "min_samples_leaf": {5, 10, 20, 100},
    #         "l2_regularization": {1e-6, 1e-3, 1e-1, 0}
    #     },

}

# BernoulliNB, MultinomialNB

from sklearn.linear_model import HuberRegressor, Lasso, SGDRegressor, ElasticNet
from sklearn.linear_model import Ridge, PassiveAggressiveRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

regressors = {
    HuberRegressor: {
        "epsilon": np.linspace(1.0, 2.0, 10),
        "alpha": alpha
    },

    Lasso: {
        "alpha": np.logspace(1, 6, 6)
    },

    SGDRegressor: {
        "loss": ["squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        "alpha": alpha
    },

    ElasticNet: {
        "alpha": np.logspace(1, 6, 6),
        "l1_ratio": np.linspace(0, 1, 4)
    },

    Ridge: {
        "alpha": alpha,
        "tol": tol,
    },

    PassiveAggressiveRegressor: {
        "C": C,
        "max_iter": [100, 300, 1000]
    },

    SVR: {
        "C": C,
        "kernel": ["linear", "poly", "rbf", "sigmoid"]
    },

    RandomForestRegressor: {
        "n_estimators": [10, 100, 200],
        "max_depth": [None, 10, 50],
        "min_samples_split": [2, 0.001, 0.01, 0.1]
    },

    ExtraTreesRegressor: {
        "n_estimators": [10, 100, 200],
        "max_depth": [None, 10, 50],
        "min_samples_split": [2, 0.001, 0.01, 0.1]
    },

    AdaBoostRegressor: {
        "n_estimators": [10, 50, 100],
        "learning_rate": [0.1, 1, 10, 100]
    },

    BaggingRegressor: {
        "n_estimators": [10, 100, 200],
        "max_samples": [1, 0.001, 0.01, 0.1],
        "max_features": [1, 0.001, 0.01, 0.1]
    },

    XGBRegressor: {
        "max_depth": [1, 6, 10, 20],
        "alpha": alpha,
    },

    KNeighborsRegressor: {
        "n_neighbors": [1, 2, 5, 10],
        "weights": ["uniform", "distance"],
    },

    MLPRegressor: {
        "solver": ["lbfgs", "sgd"],
        "alpha": alpha
    }
}

