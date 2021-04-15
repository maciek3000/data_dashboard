from ml_dashboard.analyzer import Analyzer
from ml_dashboard.features import CategoricalFeature, NumericalFeature
import pandas as pd
import pytest
import math


def test_analyzer_numeric_describe(fixture_features, numerical_features):
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
    actual_df = analyzer._create_describe_df(numerical_features).round(2)

    assert expected_df.equals(actual_df)


def test_analyzer_categorical_describe(fixture_features, categorical_features):
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
    actual_df = analyzer._create_describe_df(categorical_features).round(2)

    assert expected_df.equals(actual_df)


def test_analyzer_head_dataframe(data_classification_balanced, fixture_features, expected_raw_mapping):
    """Testing if .df_head() returns correct dataframe."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)
    cols = sorted(["Sex", "AgeGroup", "Height", "Product", "Price", "bool", "Target"], key=lambda x: x.upper())
    expected_df = df[cols].head().T

    analyzer = Analyzer(fixture_features)
    actual_df = analyzer.df_head()

    assert expected_df.equals(actual_df)


@pytest.mark.parametrize(
    ("feature", "desc", "missing", "category"),
    (
            ("Sex", "Sex of the Participant", 1.0, "cat"),
            ("AgeGroup", "Description not Available", 0.0, "cat"),
            ("Height", "Height of the Participant", 1.0, "num"),
            ("Product", "Product bought within the Transaction", 1.0, "cat"),
            ("Price", "Price of the Product", 2.0, "num"),
            ("bool", "Random Flag", 0.0, "cat"),
            ("Target", "Was the Transaction satisfactory?\nTarget Feature", 0.0, "cat")
    )
)
def test_analyzer_summary_statistics(
        data_classification_balanced, fixture_features, expected_mapping, feature, desc, missing, category
):
    """Testing if _summary_statistics() method generates correct output."""
    df = fixture_features.data()
    describe_dict = df.describe().round(4).to_dict()
    print(describe_dict)
    if category == "cat":
        expected_category = CategoricalFeature.type
    elif category == "num":
        expected_category = NumericalFeature.type
    else:
        raise

    a = Analyzer(fixture_features)

    desc_keyword = a._feature_description
    cat_keyword = a._feature_type
    missing_keyword = a._feature_missing

    expected_dict = describe_dict[feature]
    expected_dict[desc_keyword] = desc
    expected_dict[cat_keyword] = expected_category
    expected_dict[missing_keyword] = missing

    actual_dict = a.summary_statistics()[feature]

    assert actual_dict == expected_dict


@pytest.mark.parametrize(
    ("feature_name", "expected_number_of_bins"),
    (
            ("AgeGroup", 9),
            ("bool", 2),
            ("Height", 9),
            ("Price", 9),
            ("Product", 10),
            ("Target", 2),
            ("Sex", 2)
    )
)
def test_analyzer_histogram_data(fixture_features, feature_name, expected_number_of_bins):
    """Testing if ._histogram_data() method returns correct values.
        Checking only number of bins for every feature, as edges were 1) tested elsewhere 2) assuming that
        np.histogram works correctly 3) would be cumbersome to calculate all of that by hand."""

    analyzer = Analyzer(fixture_features)
    histogram_output = analyzer.histogram_data()

    # number of bins - size of one of the edges array
    actual_number_of_bins = histogram_output[feature_name][1].shape[0]

    assert actual_number_of_bins == expected_number_of_bins


@pytest.mark.parametrize(
    ("feature", "expected_result"),
    (
            ("Sex", [1.0000, -0.1583, -0.1258, -0.1597, 0.2015, 0.0298, 0.1529]),
            ("AgeGroup", [-0.1583, 1.0000, -0.1457, 0.0788, 0.0757, -0.1147, 0.0399]),
            ("Height", [-0.1258, -0.1457, 1.0000, 0.0945, -0.2027, 0.1233, 0.0780]),
            ("Product", [-0.1597, 0.0788, 0.0945, 1.0000, 0.0516, 0.0843, -0.1203]),
            ("Price", [0.2015, 0.0757, -0.2027, 0.0516, 1.0000, -0.0293, 0.0575]),
            ("bool", [0.0298, -0.1147, 0.1233, 0.0843, -0.0293, 1.0000,  0.2127]),
            ("Target", [0.1529, 0.0399, 0.0780, -0.1203, 0.0575, 0.2127, 1.0000]),
    )
)
def test_analyzer_correlation_data(fixture_features, feature, expected_result):
    """Testing if correlations between features are calculated correctly."""
    cols_in_order = ["Sex", "AgeGroup", "Height", "Product", "Price", "bool", "Target"]
    rs = 1
    analyzer = Analyzer(fixture_features)
    corr = analyzer.correlation_data_normalized(random_state=rs).loc[cols_in_order, cols_in_order]

    actual_result = corr[feature].round(4)

    assert actual_result.to_list() == expected_result


@pytest.mark.parametrize(
    ("feature", "expected_result"),
    (
            ("Sex", [1.0000, -0.2095, -0.1478, -0.1946, 0.2545, 0.0298, 0.1529]),
            ("AgeGroup", [-0.2095, 1.0000, -0.1624, 0.1263, 0.0275, -0.0765, 0.0499]),
            ("Height", [-0.1478, -0.1624, 1.0000, 0.0535, -0.1939, 0.0935, 0.0861]),
            ("Product", [-0.1946, 0.1263, 0.0535, 1.0000, -0.0555, 0.0870, -0.1697]),
            ("Price", [0.2545, 0.0275, -0.1939, -0.0555, 1.0000, -0.0917, -0.0124]),
            ("bool", [0.0298, -0.0765, 0.0935, 0.0870, -0.0917, 1.0000, 0.2127]),
            ("Target", [0.1529, 0.0499, 0.0861, -0.1697, -0.0124, 0.2127, 1.0000]),
    )
)
def test_analyzer_correlation_data_raw(fixture_features, feature, expected_result):
    """Testing if correlations between features are calculated correctly."""
    cols_in_order = ["Sex", "AgeGroup", "Height", "Product", "Price", "bool", "Target"]
    analyzer = Analyzer(fixture_features)
    corr = analyzer.correlation_data_raw().loc[cols_in_order, cols_in_order]

    actual_result = corr[feature].round(4)

    assert actual_result.to_list() == expected_result


def test_analyzer_scatter_data(
        fixture_features, data_classification_balanced, expected_raw_mapping, categorical_features
):
    """Testing if ._scatter_data() returns correct values."""
    analyzer = Analyzer(fixture_features)
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    df = pd.concat([X, y], axis=1)
    cols = sorted(["Sex", "AgeGroup", "Height", "Product", "Price", "bool", "Target"], key=lambda x: x.upper())
    df = df[cols].replace(expected_raw_mapping)

    suffix = analyzer._categorical_suffix

    for feat in categorical_features:
        df[feat + suffix] = df[feat].apply(lambda x: str(x))

    expected = df.dropna().to_dict(orient="list")
    actual = analyzer.scatter_data()

    assert expected == actual
