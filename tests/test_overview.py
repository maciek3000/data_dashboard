import pytest
from create_model.views import Overview


@pytest.mark.parametrize(
    ("expected_string",),
    (
        ("<tr><th><p>Sex<span>Sex of the Participant<br/><br/>Category - Original (Transformed)<br/>1 - Female<br/>2 - Male</span></p></th><td></td></tr>",),
        ("<tr><th>Invalid Column</th><td></td></tr>",),
        ("<tr><th><p>Target<span>Was the Transaction satisfactory?<br/>Target Feature<br/><br/>Category - Original (Transformed)<br/>1 - No<br/>2 - Yes</span></p></th><td></td></tr>",),
    )
)
def test_append_descriptions_to_table(feature_descriptor, fixture_features, test_html_table, expected_string):
    # TODO: test with >10 categories
    o = Overview("test_template", "test_css", "test_output_directory")
    actual_html = o._append_descriptions_to_features(test_html_table, fixture_features)
    print(actual_html)
    assert expected_string in actual_html
