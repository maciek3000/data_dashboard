from create_model.data_explainer import DataExplainer
import pandas as pd

def test_data_explainer_analyze_columns(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    explainer = DataExplainer(X, y)

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

    explainer = DataExplainer(X, y)
    actual_df = explainer._numeric_describe().round(2)

    _ = expected_df[expected_df != actual_df]

    assert expected_df.equals(actual_df)

def test_data_explainer_categorical_to_ordinal(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    # debugging purposes
    _ = pd.concat([X, y], axis=1)[["AgeGroup", "bool", "Product", "Sex", "Target"]]

    cat_cols = ["AgeGroup", "bool", "Product", "Sex", "Target"]
    replace_dict = {
        "Product": {
            "Oranges": 1,
            "Butter": 2,
            "Cheese": 3,
            "Bananas": 4,
            "Ketchup": 5,
            "Apples": 6,
            "Bread": 7,
            "Honey": 8,
            "Cookies": 9,
            "Eggs": 10
        },
        "Sex": {
            "Male": 2,
            "Female": 1
        },
        "AgeGroup": {
            23: 1,
            38: 2,
            48: 3,
            53: 4,
            58: 5,
            33: 6,
            18: 7,
            43: 8,
            28: 9
        },
        "bool": {
            0: 1,
            1: 2
        },
        "Target": {
            0: 1,
            1: 2
        }
    }

    concated_df = pd.concat([X, y], axis=1)
    expected_df = pd.DataFrame(concated_df[cat_cols].replace(replace_dict))

    explainer = DataExplainer(X, y)
    actual_df = explainer._categorical_to_ordinal(concated_df[cat_cols])

    assert expected_df.equals(actual_df)

def test_data_explainer_categorical_describe(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    explainer = DataExplainer(X, y)

    # debugging purposes
    # _ = explainer._categorical_to_ordinal(pd.concat([X, y], axis=1)[
    #                                           ["AgeGroup", "bool", "Product", "Sex", "Target"]]
    #                                       ).describe().T

    expected_df = pd.DataFrame({
        "count": [100.0, 98, 99.0, 99.0, 100],
        "mean": [5.05, 1.52, 5.19, 1.51, 1.60],
        "std": [2.66, 0.5, 2.76, 0.50, 0.49],
        "min": [1, 1, 1.0, 1.0, 1],
        "25%": [2, 1, 3.0, 1.0, 1],
        "50%": [5.0, 2, 5.0, 2.0, 2],
        "75%": [7.0, 2, 8.0, 2.0, 2],
        "max": [9.0, 2, 10.0, 2.0, 2],
        "missing": [0, 0.02, 0.01, 0.01, 0.0]
    }, index=["AgeGroup", "bool", "Product", "Sex", "Target"])

    actual_df = explainer._categorical_describe().round(2)

    assert expected_df.equals(actual_df)
