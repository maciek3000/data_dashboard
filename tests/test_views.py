import pytest
from create_model.views import Overview, FeatureView, append_description
from bs4 import BeautifulSoup


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
def test_overview_append_descriptions(html_test_table, header_index, description, expected_html):
    """Testing if appending description (wrapped in HTML tags) to the element works properly."""
    html_table = BeautifulSoup(html_test_table, "html.parser")
    actual_html = str(append_description(description, html_table))

    assert actual_html == expected_html


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
