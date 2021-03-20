from sklearn.metrics import accuracy_score

import os
import pandas as pd
import json

from create_model.coordinator import Coordinator
from create_model.examples.examples import iris, boston, diabetes, digits, wine, breast_cancer

if __name__ == "__main__":

    # titanic
    data_directory = os.path.join(os.getcwd(), "data", "titanic")
    output_directory = os.path.join(os.getcwd(), "output", "titanic")
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
    # X, y, descriptions = digits()  # 64 features
    # X, y, descriptions = wine()
    # X, y, descriptions = breast_cancer()  # 30 features

    coord = Coordinator(X, y, output_directory, accuracy_score, descriptions, os.getcwd())
    coord.create_html()



    # model = coord.find_model()
    #
    # predictions = model.predict(coord.transform(test_df[features]))
    # output = pd.DataFrame({'PassengerId': test_df["PassengerId"], 'Survived': predictions})
    #
    # output.to_csv(os.path.join(output_directory, "submission.csv"), index=False)
