"""Test scribbles."""
from sklearn.metrics import accuracy_score, mean_squared_error, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.impute import SimpleImputer
import numpy as np

import os
import pandas as pd
import json

from data_dashboard.dashboard import Dashboard
from data_dashboard.examples.examples import iris, boston, diabetes, digits, wine, breast_cancer

if __name__ == "__main__":

    # titanic
    data_directory = os.path.join(os.getcwd(), "data", "titanic")
    output_directory = os.path.join(os.getcwd(), "output")
    train_file = os.path.join(data_directory, "train.csv")
    test_file = os.path.join(data_directory, "test.csv")

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    target = "Survived"
    features = list(set(train_df.columns) - {target})

    X = train_df[features]
    y = train_df[target]

    descriptions = json.load(open(os.path.join(data_directory, "feature_descriptions.json")))

    # examples

    X, y, descriptions = iris()
    # X, y, descriptions = boston()
    # X, y, descriptions = diabetes()
    # X, y, descriptions = wine()
    # X, y, descriptions = breast_cancer()  # 30 features
    # X, y, descriptions = digits()

    coord = Dashboard(X, y, output_directory, feature_descriptions_dict=descriptions, random_state=42)

    # models = [SVC(C=1000.0, gamma='auto', tol=0.1, kernel="rbf"),
    #           SVC(C=1.0, gamma='auto', tol=10.0, kernel="linear"),
    #           SVC(C=10.0, kernel="linear"),
    #           SVC(C=100.0)
    #           ]

    # models = [SVR(C=1000.0, gamma='auto', tol=0.1, kernel="rbf"),
    #           SVR(C=1.0, gamma='auto', tol=10.0, kernel="linear"),
    #           SVR(C=10.0, kernel="linear"),
    #           SVR(C=100.0)
    #           ]

    models = None
    coord.create_dashboard(models=models, mode="quick")
