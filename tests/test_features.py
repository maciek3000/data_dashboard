from create_model.analyzer import Analyzer

def test_data_explainer_analyze_columns(data_classification_balanced):
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]

    explainer = Analyzer(X, y)

    cols = explainer.columns[explainer.key_cols]
    cols_wo_target = explainer.columns[explainer.key_cols_wo_target]

    assert cols[explainer.key_numerical] == ["Height", "Price"]
    assert cols[explainer.key_categorical] == ["AgeGroup", "bool", "Product", "Sex", "Target"]
    assert cols[explainer.key_date] == ["Date"]

    assert cols_wo_target[explainer.key_numerical] == ["Height", "Price"]
    assert cols_wo_target[explainer.key_categorical] == ["AgeGroup", "bool", "Product", "Sex"]
    assert cols_wo_target[explainer.key_date] == ["Date"]