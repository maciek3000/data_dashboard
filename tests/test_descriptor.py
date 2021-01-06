from create_model.descriptor import FeatureDescriptor
import pytest


def test_json_read(temp_json_file):
    expected_features = ["Sex", "AgeGroup", "Height", "Date", "Product", "Price", "bool", "Target"]
    feat_with_mappings = ["AgeGroup", "Target"]
    fd = FeatureDescriptor(temp_json_file)
    actual_json = fd.json

    assert len(actual_json) == len(expected_features)
    for feat in expected_features:
        assert feat in actual_json

    for feat in feat_with_mappings:
        assert fd._mapping in actual_json[feat]


def test_feature_get(temp_json_file, feature_descriptions):
    description = "description"
    fd = FeatureDescriptor(temp_json_file)
    for feature in feature_descriptions:
        assert fd[feature] == feature_descriptions[feature][description]


@pytest.mark.parametrize(
    ("feature",),
    (
            ("AgeGroup",),
            ("Target",),
    )
)
def test_feature_mapping(temp_json_file, feature_descriptions, feature):
    fd = FeatureDescriptor(temp_json_file)
    assert fd.feature_mapping(feature) == feature_descriptions[feature]["mapping"]


@pytest.mark.parametrize(
    ("invalid_feature",),
    (
            ("Leaves",),
            ("Wheels",),
            ("CarModel",),
    )
)
def test_keyerror_raised(temp_json_file, invalid_feature):
    fd = FeatureDescriptor(temp_json_file)
    with pytest.raises(KeyError):
        _ = fd[invalid_feature]


@pytest.mark.parametrize(
    ("nm_feature",),
    (
            ("Sex",),
            ("Height",),
            ("Date",),
    )
)
def test_no_feature_mapping(temp_json_file, nm_feature):
    fd = FeatureDescriptor(temp_json_file)
    assert fd.feature_mapping(nm_feature) is None
