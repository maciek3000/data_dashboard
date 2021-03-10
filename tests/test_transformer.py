from create_model.transformer import Transformer
import pytest


def test_transformer_analyze_data(data_classification_balanced):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]

    transformer = Transformer(X, y)

    assert False
