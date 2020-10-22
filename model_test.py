from create_model.model_finder import ModelFinder
from create_model.data_transformer import Transformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from create_model.data_explainer import DataExplainer

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

import os
import pandas as pd
import numpy as np

if __name__ == "__main__":

    data_directory = os.path.join(os.getcwd(), "data", "titanic")
    output_directory = os.path.join(os.getcwd(), "output", "titanic")
    train_file = os.path.join(data_directory, "train.csv")
    test_file = os.path.join(data_directory, "test.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    target = "Survived"

    features = list(set(train_df.columns) - set([target]))

    X = train_df[features]
    y = train_df[target]

    explainer = DataExplainer(X, y)
    explainer.create_html()

    # fig.savefig(os.path.join(os.getcwd(), "output", "titanic", "pairplot.png"))

    t = Transformer(X, y).fit()
    new_X = t.transform()

    X_test = t.transform(test_df[features])

    # imp = SimpleImputer(strategy="most_frequent")
    # imp.fit(X)
    # X = pd.DataFrame(imp.transform(X), columns=X.columns)
    #
    # X_test = test_df[features]
    # X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

    # def transform_df(dataframe):
    #     dataframe["SibSp_bool"] = np.where(dataframe["SibSp"] != 0, 1, 0)
    #     dataframe["Parch_bool"] = np.where(dataframe["Parch"] != 0, 1, 0)
    #
    #     gender_map = {"female": 1, "male": 0}
    #     dataframe["Sex_bool"] = dataframe["Sex"].map(gender_map)
    #
    #     dataframe = dataframe.drop(["SibSp", "Parch", "Sex"], axis=1)
    #
    #     return dataframe

    # X = transform_df(X)
    # X_test = transform_df(X_test)
    #
    # # fitting only to train_df
    # enc = OneHotEncoder(handle_unknown="ignore")
    # enc.fit(X["Embarked"].to_numpy().reshape(-1, 1))
    #
    # X = pd.concat([X, pd.DataFrame(enc.transform(X["Embarked"].to_numpy().reshape(-1, 1)).toarray())], axis=1)
    # X_test = pd.concat([X_test, pd.DataFrame(enc.transform(X_test["Embarked"].to_numpy().reshape(-1, 1)).toarray())], axis=1)
    #
    # X = X.drop(["Embarked"], axis=1)
    # X_test = X_test.drop(["Embarked"], axis=1)

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

    random_state = 2862

    finder = ModelFinder(models_grid, new_X, y, accuracy_score, random_state=random_state)
    model, score, params = finder.find_best_model()

    print("Model: {}\nScore: {}\nParams: {}".format(model.__name__, score, params))

    clf = model(**params)
    clf.fit(new_X, y)
    # predictions from test dataset
    predictions = clf.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_df["PassengerId"], 'Survived': predictions})

    output.to_csv(os.path.join(output_directory, "submission.csv"), index=False)