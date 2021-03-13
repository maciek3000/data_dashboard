from create_model.analyzer import Analyzer, calculate_numerical_bins, modify_histogram_edges
import pandas as pd
import pytest


@pytest.mark.parametrize(
    ("input_series", "expected_result"),
    (
            ([1, 5, 5, 6, 7, 9, 10, 10, 10, 11, 15, 17, 20], 4),  # 0.25 - 6; 0.75 - 11
            ([1, 1, 1, 1, 2, 5, 70, 90, 5, 5, 4, 3, 2, 1, 8, 8], 16),  # 0.25 - 1; 0.75 - 5.75
            ([6, 3, 4, 3, 0, 9, 5, 3, 3, 3, 8, 1, 0, 8, 0, 4, 0, 2, 9, 3, 7, 1, 0, 2, 3, 6, 9, 1, 0, 6, 1, 6, 5, 3, 4,
              3, 1, 0, 1, 2, 3, 8, 9, 1, 1, 0, 6, 2, 3, 8, 1, 1, 1, 1, 6], 3),  # 0.25 - 1; 0.75 - 6
            ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9], 6)  # 0.25 - 4; 0.75 - 6
    )
)
def test_calculate_numerical_bins(input_series, expected_result):
    """Testing if calculate_numerical_bins() correctly calculates the number of bins."""
    srs = pd.Series(input_series)
    actual_result = calculate_numerical_bins(srs)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_edges", "interval_percentage", "expected_right_edge"),
    (
            ([5, 10, 15, 20, 25, 30], 0.005, [9.875, 14.875, 19.875, 24.875, 29.875]),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1, [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.005, [1.955, 2.955, 3.955, 4.955, 5.955, 6.955, 7.955, 8.955, 9.955])
    )
)
def test_modify_histogram_edges(input_edges, interval_percentage, expected_right_edge):
    """Testing if modify_histogram_edges() correctly returns arrays for left and right edges."""
    expected_left_edge = input_edges[:-1]

    actual_left_edge, actual_right_edge = modify_histogram_edges(input_edges, interval_percentage)

    assert actual_left_edge == expected_left_edge
    assert actual_right_edge == expected_right_edge


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
