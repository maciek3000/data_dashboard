from create_model.data_transformer import Transformer
import pytest


def test_transformer_analyze_data(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    transf = Transformer(X, y)

    assert transf.numerical_columns == ["Age", "Height", "Date", "Price"]
