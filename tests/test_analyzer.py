from create_model.analyzer import Analyzer
import pandas as pd


def test_data_analyzer_numeric_describe(fixture_features, numerical_features):
    """Testing if numerical_describe dataframe returned by the Analyzer is correct."""
    expected_df = pd.DataFrame({
        "count": [99.0, 98],
        "mean": [179.98, 40],
        "std": [5.40, 30],
        "min": [165.30, 1.53],
        "25%": [176.18, 18.16],
        "50%": [179.84, 35.62],
        "75%": [183.23, 51.93],
        "max": [191.67, 131.87],
        "missing": [0.01, 0.02]
    }, index=numerical_features)

    analyzer = Analyzer(fixture_features)
    actual_df = analyzer.numerical_describe_df().round(2)

    assert expected_df.equals(actual_df)


def test_data_explainer_categorical_describe(fixture_features, categorical_features):
    """Testing if categorical_describe dataframe returned by the Analyzer is correct."""
    expected_df = pd.DataFrame({
        "count": [100.0, 100.0, 99.0, 99.0, 100],
        "mean": [4.73, 1.53, 5.34, 1.51, 1.60],
        "std": [2.55, 0.5, 2.84, 0.50, 0.49],
        "min": [1, 1, 1.0, 1.0, 1],
        "25%": [2.75, 1.0, 3.0, 1.0, 1],
        "50%": [4.5, 2, 5.0, 2.0, 2],
        "75%": [6.25, 2, 8.0, 2.0, 2],
        "max": [9.0, 2, 10.0, 2.0, 2],
        "missing": [0, 0.0, 0.01, 0.01, 0.0]
    }, index=categorical_features)

    analyzer = Analyzer(fixture_features)
    actual_df = analyzer.categorical_describe_df().round(2)

    assert expected_df.equals(actual_df)

