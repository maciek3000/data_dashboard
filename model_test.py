from sklearn.metrics import accuracy_score

import os
import pandas as pd

from create_model.analysis_coordinator import Coordinator

if __name__ == "__main__":

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

    coord = Coordinator(X, y, accuracy_score, os.getcwd())
    coord.eda()

    model = coord.find_model()

    predictions = model.predict(coord.transform(test_df[features]))
    output = pd.DataFrame({'PassengerId': test_df["PassengerId"], 'Survived': predictions})

    output.to_csv(os.path.join(output_directory, "submission.csv"), index=False)
