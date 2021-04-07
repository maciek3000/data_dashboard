import pytest
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDClassifier, SGDRegressor, Lasso

from ml_dashboard.views import Overview, FeatureView,  append_description, series_to_dict, ModelsView
from ml_dashboard.views import ModelsViewClassification


@pytest.mark.parametrize(
    ("header_index", "description", "expected_html"),
    (
            (0, "Sex of the Participant", "<span>Sex of the Participant</span>"),
            (1, "Was the Transaction satisfactory?\nTarget Feature", "<span>Was the Transaction "
                                                                     "satisfactory?<br/>Target Feature</span>"),
            (2, "Price of the Product", "<span>Price of the Product</span>"),
            (3, "Description not\n Available", "<span>Description not<br/> Available</span>"),
            (4, "Height of the Participant", "<span>Height of the Participant</span>"),
            (5, "Product bought within the Transaction", "<span>Product bought within the Transaction</span>"),
            (6, "Random Flag", "<span>Random Flag</span>")
    )
)
def test_append_descriptions(html_test_table, header_index, description, expected_html):
    """Testing if appending description (wrapped in HTML tags) to the element works properly."""
    html_table = BeautifulSoup(html_test_table, "html.parser")
    actual_html = str(append_description(description, html_table))

    assert actual_html == expected_html


@pytest.mark.parametrize(
    ("param_series", "expected_result"),
    (
            (pd.Series({"1": {'C': 1.0, 'alpha': 0.001}, "2": {'max_depth': 10.0, 'criterion': 'gini'}}),
             {"1": "'C': 1.0\n'alpha': 0.001", "2": "'max_depth': 10.0\n'criterion': 'gini'"}),
            (pd.Series({
                "test1": {"C": 1.0, "alpha": 0.001, "depth": "10"},
                "test2": {"aa": 1.0, "bb": "cc", "test": "test"},
                "test3": {"test1": 10.0, "C": 0.001, "alpha": 1000.0}
            }),
             {
                 "test1": "'C': 1.0\n'alpha': 0.001\n'depth': '10'",
                 "test2": "'aa': 1.0\n'bb': 'cc'\n'test': 'test'",
                 "test3": "'test1': 10.0\n'C': 0.001\n'alpha': 1000.0"
             })
    )
)
def test_series_to_dict(param_series, expected_result):
    actual_result = series_to_dict(param_series)
    assert actual_result == expected_result


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
    o = Overview("test_template", "test_css", "test_output_directory", 10, "test-description")
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
    o = Overview("test_template", "test_css", "test_output_directory", 10, "test-description")
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

    o = Overview("test_template", "test_css", "test_output_directory", 5, test_description)  # max_categories == 5

    expected_mapping["Price"] = None
    expected_mapping["Height"] = None
    actual_html = o._stylize_html_table(html_test_table, expected_mapping, descriptions)

    assert actual_html == expected_html


# @pytest.mark.parametrize(
#     ("input_df", "expected_result"),
#     (
#             (
#                     pd.DataFrame(data={"test1": [1, 2, 3, 4], "test2": ["a", "b", "c", "d"]},
#                                  index=["ind1", "ind2", "ind3", "ind4"]),
#                     '<table><thead><tr><th></th><th>test1</th><th>test2</th></tr></thead>'
#                     '<tbody>'
#                     '<tr><th>ind1</th><td>1</td><td>a</td></tr>'
#                     '<tr><th>ind2</th><td>2</td><td>b</td></tr>'
#                     '<tr><th>ind3</th><td>3</td><td>c</td></tr>'
#                     '<tr><th>ind4</th><td>4</td><td>d</td></tr>'
#                     '</tbody></table>'
#             ),
#             (
#                 pd.DataFrame(data={
#                     "model1": [1, "test"],
#                     "model2": [2, True],
#                     "model3": [3, "None"],
#                     "model4": [4, "aaa"]
#                     },
#                     index=["index1", "index2"]
#                 ),
#                 '<table><thead><tr><th></th>'
#                 '<th>model1</th><th>model2</th><th>model3</th><th>model4</th>'
#                 '</tr></thead>'
#                 '<tbody>'
#                 '<tr><th>index1</th><td>1</td><td>2</td><td>3</td><td>4</td></tr>'
#                 '<tr><th>index2</th><td>test</td><td>True</td><td>None</td><td>aaa</td></tr>'
#                 '</tbody></table>'
#             )
#     )
# )
# def test_df_to_html_table(input_df, expected_result):
#     actual_result = df_to_html_table(input_df)
#     assert actual_result == expected_result


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
    o = Overview("test_template", "test_css", "test_output_directory", 5, "test-description")
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
def test_feature_view_create_features_menu(input_features):
    """Testing if Features menu is created properly given the input features."""
    title = "<div>Title</div>"
    single_feature = "<span>{}. {}</span>"
    fv = FeatureView("test_template", "test_css", "test_html")
    fv._feature_menu_header = title
    fv._feature_menu_single_feature = single_feature
    actual_result = fv._create_features_menu(input_features)

    expected_result = title
    i = 0
    for feat in input_features:
        expected_result += single_feature.format(i, feat)
        i += 1

    assert actual_result == expected_result


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
    mv = ModelsView("template", "test_css", "test_js", "params", "test-class")
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
    mv = ModelsViewClassification("template", "test_css", "test_js", "params", "test-class")
    mv._confusion_matrices_single_matrix_table = expected_class
    actual_result = mv._single_confusion_matrix_html(input_array)

    assert expected_string in actual_result
    assert expected_class in actual_result


@pytest.mark.parametrize(
    ("input_tuple",),
    (
            ([
                    (Lasso(), np.array(([1, 2], [3, 4]))),
            ],),
            ([
                    (Lasso(), np.array(([10, 20], [40, 50]))),
                    (LinearRegression(), np.array(([70, 70], [1, 1]))),
                    (SGDRegressor(), np.array(([0, 0], [0, 0]))),
                    (SGDClassifier(), np.array(([1, 1,], [1, 1])))
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
    mv = ModelsViewClassification("template", "test_css", "test_js", "params", "test-class")

    mv._first_model_class = first_model
    mv._other_model_class = other_model
    mv._confusion_matrices_single_matrix_title = title_class
    mv._confusion_matrices_single_matrix = matrix_class

    actual_results = mv._confusion_matrices(input_tuple)

    parsed = BeautifulSoup(actual_results, "html.parser")
    titles = parsed.select("." + title_class)
    matrices = parsed.select("." + matrix_class)

    for actual_title, expected_tuple in zip(titles, input_tuple):
        assert actual_title.string == expected_tuple[0].__class__.__name__

    assert first_model in matrices[0]["class"]
    for matrix in matrices[1:]:
        assert other_model in matrix["class"]
