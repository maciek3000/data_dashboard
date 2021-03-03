from create_model.analyzer import Analyzer
import pandas as pd

def test_data_explainer_analyze_columns(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    explainer = Analyzer(X, y)

    cols = explainer.columns[explainer.key_cols]
    cols_wo_target = explainer.columns[explainer.key_cols_wo_target]

    assert cols[explainer.key_numerical] == ["Height", "Price"]
    assert cols[explainer.key_categorical] == ["AgeGroup", "bool", "Product", "Sex", "Target"]
    assert cols[explainer.key_date] == ["Date"]

    assert cols_wo_target[explainer.key_numerical] == ["Height", "Price"]
    assert cols_wo_target[explainer.key_categorical] == ["AgeGroup", "bool", "Product", "Sex"]
    assert cols_wo_target[explainer.key_date] == ["Date"]

def test_data_explainer_numeric_describe(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    # debugging purposes
    # _ = X[["Height", "Price"]].describe().T

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
    }, index=["Height", "Price"])

    explainer = Analyzer(X, y)
    actual_df = explainer._numeric_describe().round(2)

    _ = expected_df[expected_df != actual_df]

    assert expected_df.equals(actual_df)

def test_data_explainer_categorical_mapping(test_data_classification_balanced, expected_mapping):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    # debugging purposes
    # _ = pd.concat([X, y], axis=1)[["AgeGroup", "bool", "Product", "Sex", "Target"]]

    explainer = Analyzer(X, y)
    actual_mapping = explainer._create_categorical_mapping()

    assert actual_mapping == expected_mapping

def test_data_explainer_categorical_describe(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    explainer = Analyzer(X, y)

    # debugging purposes
    _ = pd.concat([X, y], axis=1)[["AgeGroup", "bool", "Product", "Sex", "Target"]].replace(explainer._create_categorical_mapping()).describe().T

    expected_df = pd.DataFrame({
        "count": [100.0, 98, 99.0, 99.0, 100],
        "mean": [3.73, 0.52, 4.34, 0.51, 0.60],
        "std": [2.55, 0.5, 2.84, 0.50, 0.49],
        "min": [0, 0, 0.0, 0.0, 0],
        "25%": [1.75, 0.0, 2.0, 0.0, 0],
        "50%": [3.5, 1, 4.0, 1.0, 1],
        "75%": [5.25, 1, 7.0, 1.0, 1],
        "max": [8.0, 1, 9.0, 1.0, 1],
        "missing": [0, 0.02, 0.01, 0.01, 0.0]
    }, index=["AgeGroup", "bool", "Product", "Sex", "Target"])

    actual_df = explainer._categorical_describe().round(2)

    assert expected_df.equals(actual_df)
