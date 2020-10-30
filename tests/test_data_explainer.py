from create_model.data_explainer import DataExplainer
import pandas as pd

def test_data_explainer_analyze_columns(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    explainer = DataExplainer(X, y)

    cols = explainer.columns[explainer.key_cols]
    cols_wo_target = explainer.columns[explainer.key_cols_wo_target]

    assert cols[explainer.key_numerical] == ["Age", "bool", "Height", "Price", "Target"]
    assert cols[explainer.key_categorical] == ["Product", "Sex"]
    assert cols[explainer.key_date] == ["Date"]

    assert cols_wo_target[explainer.key_numerical] == ["Age", "bool", "Height", "Price"]
    assert cols_wo_target[explainer.key_categorical] == ["Product", "Sex"]
    assert cols_wo_target[explainer.key_date] == ["Date"]

def test_data_explainer_numeric_describe(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    # debugging purposes
    # _ = pd.concat([X, y], axis=1)[["Age", "bool", "Height", "Price", "Target"]].describe().T

    expected_df = pd.DataFrame({
        "count": [100.0, 98, 99, 98, 100],
        "mean": [36.65, 0.52, 179.98, 40, 0.60],
        "std": [12.75, 0.5, 5.40, 30, 0.49],
        "min": [18, 0, 165.30, 1.53, 0],
        "25%": [26.75, 0, 176.18, 18.16, 0],
        "50%": [35.5, 1, 179.84, 35.62, 1],
        "75%": [44.25, 1, 183.23, 51.93, 1],
        "max": [58, 1, 191.67, 131.87, 1],
        "missing": [0, 0.02, 0.01, 0.02, 0]
    }, index=["Age", "bool", "Height", "Price", "Target"])

    explainer = DataExplainer(X, y)
    actual_df = explainer._numeric_describe().round(2)

    assert expected_df.equals(actual_df)

def test_data_explainer_categorical_to_ordinal(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    # debugging purposes
    # _ = X[["Product", "Sex"]]

    cat_cols = ["Product", "Sex"]
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
        }
    }

    expected_df = pd.DataFrame(X[cat_cols].replace(replace_dict))

    explainer = DataExplainer(X, y)
    actual_df = explainer._categorical_to_ordinal(X[cat_cols])

    print(expected_df[(expected_df["Product"] != actual_df["Product"])])

    assert expected_df.equals(actual_df)

def test_data_explainer_categorical_describe(test_data_classification_balanced):
    X = test_data_classification_balanced[0]
    y = test_data_classification_balanced[1]

    explainer = DataExplainer(X, y)

    # debugging purposes
    _ = explainer._categorical_to_ordinal(X[["Product", "Sex"]]).describe().T

    expected_df = pd.DataFrame({
        "count": [99.0, 99.0],
        "mean": [5.19, 1.51],
        "std": [2.76, 0.50],
        "min": [1.0, 1.0],
        "25%": [3.0, 1.0],
        "50%": [5.0, 2.0],
        "75%": [8.0, 2.0],
        "max": [10.0, 2.0],
        "missing": [0.01, 0.01]
    }, index=["Product", "Sex"])

    actual_df = explainer._categorical_describe().round(2)

    assert expected_df.equals(actual_df)
