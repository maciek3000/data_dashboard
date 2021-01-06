import pytest
from create_model.overview import Overview

@pytest.mark.parametrize(
    ("feature", "expected_result"),
    (
            ("Sex", {"Female": "0", "Male": "1"}),
            ("AgeGroup", {
                "Between 18 and 22": "18 (0)",
                "Between 23 and 27": "23 (1)",
                "Between 28 and 32": "28 (2)",
                "Between 33 and 37": "33 (3)",
                "Between 38 and 42": "38 (4)",
                "Between 43 and 47": "43 (5)",
                "Between 48 and 52": "48 (6)",
                "Between 53 and 57": "53 (7)",
                "Between 58 and 62": "58 (8)"
            }),
            ("Height", ""),
            ("Date", ""),
            ("Product", {
                "Apples": "0",
                "Bananas": "1",
                "Bread": "2",
                "Butter": "3",
                "Cheese": "4",
                "Cookies": "5",
                "Eggs": "6",
                "Honey": "7",
                "Ketchup": "8",
                "Oranges": "9"
            }),
            ("Price", ""),
            ("bool", {"0": "0", "1": "1"}),
            ("Target", {"No": "0 (0)", "Yes": "1 (1)"})
    )
)
def test_consolidate_mappings(feature_descriptor, expected_mapping, feature, expected_result):
    overview = Overview("test_template", "test_css", feature_descriptor, expected_mapping)
    actual_mapping = overview._consolidate_mappings(feature)

    assert actual_mapping == expected_result

@pytest.mark.parametrize(
    ("expected_string",),
    (
        ("<tr><th><p>Sex<span>Sex of the Participant<br/><br/>Category - Original (Transformed)<br/>Female - 0<br/>Male - 1</span></p></th><td></td></tr>",),
        ("<tr><th>Invalid Column</th><td></td></tr>",),
        ("<tr><th><p>Target<span>Was the Transaction satisfactory?<br/>Target Feature<br/><br/>Category - Original (Transformed)<br/>No - 0 (0)<br/>Yes - 1 (1)</span></p></th><td></td></tr>",),
    )
)
def test_append_descriptions_to_table(feature_descriptor, expected_mapping, test_html_table, expected_string):
    # TODO: test with >10 categories
    overview = Overview("test_template", "test_css", feature_descriptor, expected_mapping)
    actual_html = overview._Overview__append_description(test_html_table)
    print(actual_html)
    assert expected_string in actual_html
