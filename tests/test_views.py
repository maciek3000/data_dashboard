import pytest
from create_model.views import Overview, FeatureView


@pytest.mark.parametrize(
    ("expected_string",),
    (
        ("<tr><th><p>Sex<span>Sex of the Participant<br/><br/>Category - Original (Transformed)<br/>1 - Female<br/>2 - Male</span></p></th><td></td></tr>",),
        ("<tr><th><p>Price<span></span></p></th><td></td></tr>",),
        ("<tr><th><p>Target<span>Was the Transaction satisfactory?<br/>Target Feature<br/><br/>Category - Original (Transformed)<br/>1 - No<br/>2 - Yes</span></p></th><td></td></tr>",),
    )
)
def test_stylize_html_table(feature_descriptor, fixture_features, expected_mapping, test_html_table, expected_string):
    # TODO: test with >10 categories
    o = Overview("test_template", "test_css", "test_output_directory")
    descriptions = fixture_features.descriptions()
    descriptions["Price"] = ""

    num_map = {
        "Price": None,
        "Height": None
    }

    expected_mapping.update(num_map)

    actual_html = o._stylize_html_table(test_html_table, expected_mapping, descriptions)
    print(actual_html)
    assert expected_string in actual_html


def test_overview_append_descriptions():
    assert False

@pytest.mark.parametrize(
    ("input_features",),
    (
            (["Feature1", "Feature2", "Feature 3"],),
            (["zZz Feature", "oblabla", "randomrandom"],),
            (["Feature3", "Feature1", "Feature2", "Feature5", "Feature8"],)
    )
)
def test_featureview_create_features_menu(input_features):
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
