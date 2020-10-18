from create_model.data_transformer import Transformer
import pytest


def test_transformer_analyze_data(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    transf = Transformer(X, y)

    assert set(transf.numerical_columns) == set(["Age", "Height", "Price"])
    assert set(transf.categorical_columns) == set(["Sex", "Product", "bool"])
    assert set(transf.date_columns) == set(["Date"])
