import pytest
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from bokeh.plotting import figure

from data_dashboard.views import Overview, FeatureView, ModelsView, ModelsViewClassification


@pytest.mark.parametrize(
    ("header_index", "expected_html"),
    (
            (0, "Sex<br/><br/>{header}<br/>1 - Female<br/>2 - Male</th><td>"),
            (1, "Target<br/><br/>{header}<br/>1 - No<br/>2 - Yes</th><td>"),
            (2, "Price</th><td>"),
            (3, "AgeGroup<br/><br/>{header}<br/>1 - Between 18 and 22<br/>2 - Between 23 and 27<br/>3 - Between 28 "
                "and 32<br/>4 - Between 33 and 37<br/>5 - Between 38 and 42<br/>6 - Between 43 and 47<br/>7 - Between "
                "48 and 52<br/>8 - Between 53 and 57<br/>9 - Between 58 and 62</th><td>"),
            (4, "Height</th><td>"),
            (5, "Product<br/><br/>{header}<br/>1 - Apples<br/>2 - Bananas<br/>3 - Bread<br/>4 - Butter<br/>5 - "
                "Cheese<br/>6 - Cookies<br/>7 - Eggs<br/>8 - Honey<br/>9 - Ketchup<br/>10 - Oranges</th><td>"),
            (6, "bool<br/><br/>{header}<br/>1 - False<br/>2 - True</th><td>")
    )
)
def test_overview_append_mappings(html_test_table, header_index, fixture_features, expected_html):
    """Testing if appending mappings (wrapped in HTML) to the element works properly."""
    html_table = BeautifulSoup(html_test_table, "html.parser")
    headers = html_table.table.select("table tbody tr th")
    mapping = fixture_features.mapping()[headers[header_index].string]
    o = Overview("test_template", "test_css", "test-description")
    o._append_mapping(headers[header_index], mapping, html_table)

    header = o._mapping_title
    assert expected_html.format(header=header) in str(html_table)


@pytest.mark.parametrize(
    ("header_index", "expected_html"),
    (
            (3, "AgeGroup<br/><br/>{header}<br/>1 - Between 18 and 22<br/>2 - Between 23 and 27<br/>3 - Between 28 "
                "and 32<br/>4 - Between 33 and 37<br/>5 - Between 38 and 42<br/>6 - Between 43 and 47<br/>7 - Between "
                "48 and 52<br/>8 - Between 53 and 57<br/>9 - Between 58 and 62<br/>10 - Test Value<br/>"
                "{footer}</th><td>"),
            (5, "Product<br/><br/>{header}<br/>1 - Apples<br/>2 - Bananas<br/>3 - Bread<br/>4 - Butter<br/>5 - "
                "Cheese<br/>6 - Cookies<br/>7 - Eggs<br/>8 - Honey<br/>9 - Ketchup<br/>10 - Test Value<br/>"
                "{footer}</th><td>")
    )
)
def test_overview_append_mappings_more_than_limit(html_test_table, fixture_features, header_index, expected_html):
    """Testing if, when the amount of mapped categories exceeds the limit, then the HTML mappings are chopped off
    appropriately. """
    html_table = BeautifulSoup(html_test_table, "html.parser")
    headers = html_table.table.select("table tbody tr th")
    mapping = fixture_features.mapping()[headers[header_index].string]
    mapping[10] = "Test Value"
    mapping[11] = "Test Value"
    o = Overview("test_template", "test_css", "test-description")
    o._append_mapping(headers[header_index], mapping, html_table)

    header = o._mapping_title
    footer = o._too_many_categories.format(10)

    assert expected_html.format(header=header, footer=footer) in str(html_table)


def test_stylize_html_table(html_test_table, expected_mapping, fixture_features):
    """Testing if the ._stylize_html_table() function creates correct HTML output."""
    # text is 'dedented' to match the output provided by the function.
    expected_html = """
<table>
<thead><tr><th></th><th></th></tr></thead>
<tbody>
<tr><th><p class="{test_description}">Sex<span>Sex of the Participant<br/><br/>{header}<br/>1 - Female<br/>2 - Male</span></p></th><td></td></tr>
<tr><th><p class="{test_description}">Target<span>Was the Transaction satisfactory?<br/>Target Feature<br/><br/>{header}<br/>1 - No<br/>2 - Yes</span></p></th><td></td></tr>
<tr><th><p class="{test_description}">Price<span>Price of the Product</span></p></th><td></td></tr>
<tr><th><p class="{test_description}">AgeGroup<span>Description not Available<br/><br/>{header}<br/>1 - Between 18 and 22<br/>2 - Between 23 and 27<br/>3 - Between 28 and 32<br/>4 - Between 33 and 37<br/>5 - Between 38 and 42<br/>{footer}</span></p></th><td></td></tr>
<tr><th><p class="{test_description}">Height<span>Height of the Participant</span></p></th><td></td></tr>
<tr><th><p class="{test_description}">Product<span>Product bought within the Transaction<br/><br/>{header}<br/>1 - Apples<br/>2 - Bananas<br/>3 - Bread<br/>4 - Butter<br/>5 - Cheese<br/>{footer}</span></p></th><td></td></tr>
<tr><th><p class="{test_description}">bool<span>Random Flag<br/><br/>{header}<br/>1 - False<br/>2 - True</span></p></th><td></td></tr>
</tbody>
</table>
"""

    descriptions = fixture_features.descriptions()
    header = "Category - Original"
    footer = "(...) Showing only first 5 categories"
    test_description = "test-description"
    expected_html = expected_html.format(header=header, footer=footer, test_description=test_description)

    o = Overview("test_template", "test_css", test_description)  # max_categories == 5
    o._max_categories_limit = 5

    expected_mapping["Price"] = None
    expected_mapping["Height"] = None
    actual_html = o._stylize_html_table(html_test_table, expected_mapping, descriptions)

    assert actual_html == expected_html


@pytest.mark.parametrize(
    ("input_list", "expected_string"),
    (
            (["test1", "Test2"], "<ul><li>test1</li><li>Test2</li></ul>"),
            (["feature 2", "feature 1", "aaaa"], "<ul><li>feature 2</li><li>feature 1</li><li>aaaa</li></ul>"),
            (["Test", "test", "TEST"], "<ul><li>Test</li><li>test</li><li>TEST</li></ul>")
    )
)
def test_overview_unused_features_html(input_list, expected_string):
    """Testing if creating HTML output of unused features works properly."""
    o = Overview("test_template", "test_css", "test-description")
    actual_html = o._unused_features_html(input_list)

    assert actual_html == expected_string


@pytest.mark.parametrize(
    ("input_features",),
    (
            (["Feature1", "Feature2", "Feature 3"],),
            (["zZz Feature", "oblabla", "randomrandom"],),
            (["Feature3", "Feature1", "Feature2", "Feature5", "Feature8"],)
    )
)
def test_features_view_create_features_menu(input_features):
    """Testing if Features menu is created properly given the input features."""

    fv = FeatureView("test_template", "test_css", "test_html", "test-target", {})
    title_template = fv._feature_menu_header
    fv._menu_single_feature_class = "test-class"
    actual_result = fv._create_features_menu(input_features)

    expected_result = title_template
    i = 0
    for feat in input_features:
        expected_result += "<div class='test-class'><span>{:03}. {}</span></div>".format(i, feat)
        i += 1

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("target_name", "expected_feature_text"),
    (
            ("Feature0", "000. Feature0"),
            ("Feature1", "001. Feature1"),
            ("Feature2", "002. Feature2"),
            ("Feature3", "003. Feature3"),
    )
)
def test_features_view_create_features_menu_target_name(target_name, expected_feature_text):
    """Testing if target-class is being added to the output of features_menu HTML where feature name is a target at
    the same time."""
    features = ["Feature0", "Feature1", "Feature2", "Feature3"]
    fv = FeatureView("test_template", "test_css", "test_html", target_name, {})
    fv._menu_single_feature_class = "test-class"
    fv._menu_target_feature_class = "test-target-class"
    actual_result = fv._create_features_menu(features)
    expected_text = "<div class='test-class test-target-class'><span>{text}</span></div>".format(text=expected_feature_text)
    assert expected_text in actual_result
    assert actual_result.count("test-target-class") == 1


@pytest.mark.parametrize(
    ("input_series", "input_df", "expected_result"),
    (
            (
                pd.Series([1, 2, 3], name="test1"),
                pd.DataFrame(data={"a": ["a", "b", "c"], "b": ["d", "e", "f"]}),
                """<div class='test-subtitle-class'>test-title</div><table border='1' class='dataframe'>
                <thead>
                <trstyle='text-align:right;'><th>test_prefix-test1</th><th>a</th><th>b</th></tr></thead>
                <tbody>
                <tr><td>1</td><td>a</td><td>d</td></tr>
                <tr><td>2</td><td>b</td><td>e</td></tr>
                <tr><td>3</td><td>c</td><td>f</td></tr>
                </tbody>
                </table>              
                """
            ),
            (
                pd.Series([1, 2, 3, 4], name="test2"),
                pd.DataFrame(data={"e": ["z", "y", "x", "w"]}),
                """<div class='test-subtitle-class'>test-title</div><table border='1' class='dataframe'>
                <thead>
                <trstyle='text-align:right;'><th>test_prefix-test2</th><th>e</th></tr></thead>
                <tbody>
                <tr><td>1</td><td>z</td></tr>
                <tr><td>2</td><td>y</td></tr>
                <tr><td>3</td><td>x</td></tr>
                <tr><td>4</td><td>w</td></tr>
                </tbody></table>"""
            )
    )
)
def test_features_view_transformed_dataframe_html(input_series, input_df, expected_result):
    """Testing if transformed_dataframe_html() method creates correct HTML output."""
    test_subtitle = "test-subtitle-class"
    test_title = "test-title"
    prefix = "test_prefix-"

    fv = FeatureView("test_template", "test_css", "test_html", "test-target", {})
    fv._transformed_feature_subtitle_div = test_subtitle
    fv._transformed_feature_transformed_df_title = test_title
    fv._transformed_feature_original_prefix = prefix
    actual_result = fv._transformed_dataframe_html(input_series, input_df)

    actual_result = actual_result.replace(" ", "").replace("\n", "").replace('"', "'")
    expected_result = expected_result.replace(" ", "").replace("\n", "")

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_transformers", "expected_result"),
    (
            (
                ["Transformer1(test='test')", "Transformer2(random_state=1)"],
                """<div class='test-transformer-list'>
                    <div class='test-subtitle'>Test Title For Transformers</div>
                    <div>
                        <div class='test-single-tr'>Transformer1(test='test')</div>
                        <div class='test-single-tr'>Transformer2(random_state=1)</div>
                    </div>
                    </div>"""
            ),
            (
                ["TestTransformer", "TestTransformer", "TestTransformer", "TestTransformer"],
                """<div class='test-transformer-list'>
                    <div class='test-subtitle'>Test Title For Transformers</div>
                    <div>
                        <div class='test-single-tr'>TestTransformer</div>
                        <div class='test-single-tr'>TestTransformer</div>
                        <div class='test-single-tr'>TestTransformer</div>
                        <div class='test-single-tr'>TestTransformer</div>
                    </div>
                    </div>"""
            ),
            (
                ["Test(test='test')"],
                """<div class='test-transformer-list'>
                    <div class='test-subtitle'>Test Title For Transformers</div>
                    <div>
                        <div class='test-single-tr'>Test(test='test')</div>
                    </div>
                    </div>"""
            )
    )
)
def test_features_view_transformers_html(input_transformers, expected_result):
    """Testing if transformers_html() method returns correct HTML output based on provided list of transformers."""
    fv = FeatureView("test_template", "test_css", "test_html", "test-target", {})
    fv._transformed_feature_single_transformer = "test-single-tr"
    fv._transformed_feature_transformer_list = "test-transformer-list"
    fv._transformed_feature_transformers_title = "Test Title For Transformers"
    fv._transformed_feature_subtitle_div = "test-subtitle"

    actual_result = fv._transformers_html(input_transformers)

    actual_result = actual_result.replace(" ", "").replace("\n", "").replace('"', "'")
    expected_result = expected_result.replace(" ", "").replace("\n", "")

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_feature",),
    (
            ("AgeGroup",),
            ("bool",),
            ("Height",),
            ("Price",),
            ("Product",),
            ("Sex",),
            ("Target",)
    )
)
def test_features_view_transformed_features_divs(data_classification_balanced, transformed_classification_data, transformer_classification_fitted, numerical_features, input_feature):
    """Testing if transformed_features_divs() method creates correct HTML output with appropriate classes set
    to their respective divs."""
    # setting up necessary objects
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1).drop(["Date"], axis=1)
    tr_X = pd.DataFrame(transformed_classification_data[0].toarray(), columns=transformer_classification_fitted.transformed_columns())
    tr_y = pd.Series(transformed_classification_data[1], name="Target")
    transformed_df = pd.concat([tr_X, tr_y], axis=1)
    transformations = transformer_classification_fitted.transformations()
    transformations["Target"] = (transformer_classification_fitted.y_transformations(), ["Target"])
    numerical_features = transformer_classification_fitted.numerical_features

    fv = FeatureView("test_template", "test_css", "test_html", "Target", {})
    fv._first_feature_transformed = "test-chosen-feature"
    fv._transformed_feature_div = "test-div"
    fv._transformed_feature_plots_grid = "test-grid"
    mock_plots = {feature: figure() for feature in numerical_features}
    actual_result = fv._transformed_features_divs(df.head(), transformed_df.head(), transformations, numerical_features, mock_plots, input_feature)

    expected_str = """<div class="test-div test-chosen-feature" id="{feature}">""".format(feature=input_feature)

    assert expected_str in actual_result
    assert actual_result.count("test-chosen-feature") == 1

    if input_feature in numerical_features:
        assert "test-grid" in actual_result
        assert actual_result.count("test-grid") == len(numerical_features)


@pytest.mark.parametrize(
    ("input_df", "expected_row_number"),
    (
            (
                    pd.DataFrame(
                        data={
                            "col1": [1, 2, 3, 4, 5],
                            "col2": [6, 7, 8, 9, 10],
                            "params": [{1: 1}, {2: 2}, {3: 3}, {4: 4}, {5: 5}]
                        },
                        index=["index1", "index2", "index3", "index4", "index5"]
                    ), 5),
            (
                    pd.DataFrame(
                        data={
                            "col1": [0.1, 0.2],
                            "col2": [0.6, 0.7],
                            "col3": ["test", "test"],
                            "col4": [True, False],
                            "params": [{"A": "a"}, {"B": "b"}]
                        },
                        index=["index1", "index2"]
                    ),
                    2
            )
    )
)
def test_model_view_results_table(input_df, expected_row_number):
    """Testing if html output produced by results_table method properly assigns css classes to table rows."""
    mv = ModelsView("template", "test_css", "params", "test-class")
    first_row_class = mv._first_model_class
    middle_row_class = mv._other_model_class

    actual_result = mv._models_result_table(input_df)
    table = BeautifulSoup(actual_result, "html.parser")
    rows = table.select("table tbody tr")

    assert len(rows) == expected_row_number
    assert rows[0]["class"] == [first_row_class]
    for row in rows[1:-1]:
        assert row["class"] == [middle_row_class]
    with pytest.raises(KeyError):
        _ = rows[-1]["class"]


@pytest.mark.parametrize(
    ("input_array", "expected_string"),
    (
            (np.array(([1, 0], [2, 3])), "<tr><th>Actual Negative</th><td>1</td><td>0</td></tr>" \
                                         "<tr><th>Actual Positive</th><td>2</td><td>3</td></tr>"
             ),

            (np.array(([15, 20], [100, 80])), "<tr><th>Actual Negative</th><td>15</td><td>20</td></tr>" \
                                              "<tr><th>Actual Positive</th><td>100</td><td>80</td></tr>")
    )
)
def test_models_view_classification_single_matrix_table(input_array, expected_string):
    """Testing if confusion matrix html table is created correctly."""
    expected_class = "test-class"
    mv = ModelsViewClassification("template", "test_css", "params", "test-class")
    mv._confusion_matrices_single_matrix_table = expected_class
    actual_result = mv._single_confusion_matrix_html(input_array)

    assert expected_string in actual_result
    assert expected_class in actual_result


@pytest.mark.parametrize(
    ("input_tuple",),
    (
            ([
                    ("Lasso", np.array(([1, 2], [3, 4]))),
            ],),
            ([
                    ("Lasso", np.array(([10, 20], [40, 50]))),
                    ("Lasso", np.array(([70, 70], [1, 1]))),
                    ("SGDRegressor", np.array(([0, 0], [0, 0]))),
                    ("SGDClassifier", np.array(([1, 1,], [1, 1])))
            ],),
    )
)
def test_models_view_classification_confusion_matrices(input_tuple):
    """Testing if the confusion matrices html is created correctly and classes are assigned to elements
    appropriately."""
    first_model = "test-first-model"
    other_model = "test-other-model"
    title_class = "test-title-class"
    matrix_class = "test-matrix-class"
    mv = ModelsViewClassification("template", "test_css", "params", "test-class")

    mv._first_model_class = first_model
    mv._other_model_class = other_model
    mv._confusion_matrices_single_matrix_title = title_class
    mv._confusion_matrices_single_matrix = matrix_class

    actual_results = mv._confusion_matrices(input_tuple)

    parsed = BeautifulSoup(actual_results, "html.parser")
    titles = parsed.select("." + title_class)
    matrices = parsed.select("." + matrix_class)

    for actual_title, expected_tuple in zip(titles, input_tuple):
        assert actual_title.string == expected_tuple[0]

    assert first_model in matrices[0]["class"]
    for matrix in matrices[1:]:
        assert other_model in matrix["class"]
