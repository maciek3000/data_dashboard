from create_model.data_transformer import Transformer
import pytest


def test_transformer_analyze_data(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    transformer = Transformer(X, y)

    assert False
