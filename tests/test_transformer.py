from create_model.transformer import Transformer
import pytest


def test_transformer_analyze_data(data_classification_balanced, fixture_features):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]

    transformer = Transformer(X, y, fixture_features.numerical_features(), fixture_features.categorical_features())

    assert True
