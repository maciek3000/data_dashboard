from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.svm import SVC

import os
import pandas as pd
import json

from ml_dashboard.coordinator import Coordinator
from ml_dashboard.examples.examples import iris, boston, diabetes, digits, wine, breast_cancer

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

    # X, y, descriptions = iris()
    # X, y, descriptions = boston()
    # X, y, descriptions = diabetes()
    # X, y, descriptions = wine()
    # X, y, descriptions = breast_cancer()  # 30 features

    coord = Coordinator(X, y, output_directory, accuracy_score, descriptions, os.getcwd(), random_state=42)
    model = coord.search_and_fit()
    coord.create_dashboard()
    # output = coord.quick_find()
    # print("\n".join(map(lambda x: x[0] + ": " + str(x[1]), output)))

    model = SVC(C=1000.0, gamma='auto', tol=0.1, kernel="rbf")
    #ml = coord.search_and_fit(mode="quick", scoring=accuracy_score)

    #predictions = coord.predict(test_df[features])
    # results = coord.model_finder._quicksearch_results

    #output = pd.DataFrame({'PassengerId': test_df["PassengerId"], 'Survived': predictions})
    #print(ml)
    #print(coord.model_finder._gridsearch_results.to_markdown())
    #print(output)






    #output.to_csv(os.path.join(output_directory, "submission.csv"), index=False)
