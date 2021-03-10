from create_model.descriptor import FeatureDescriptor
import pytest


def test_feature_get(feature_descriptions):
    fd = FeatureDescriptor(feature_descriptions)
    for feature in feature_descriptions:
        assert fd[feature] == feature_descriptions[feature]


@pytest.mark.parametrize(
    ("feature",),
    (
            ("AgeGroup",),
            ("Target",),
    )
)
def test_feature_mapping(feature_descriptions, feature):
    mapping = FeatureDescriptor._mapping
    fd = FeatureDescriptor(feature_descriptions)
    assert fd.mapping(feature) == feature_descriptions[feature][mapping]


def test_feature_category(feature_descriptions):
    category = FeatureDescriptor._category
    expected_feature = "Target"
    fd = FeatureDescriptor(feature_descriptions)
    assert fd.category(expected_feature) == feature_descriptions[expected_feature][category]


@pytest.mark.parametrize(
    ("invalid_feature",),
    (
            ("Leaves",),
            ("Wheels",),
            ("CarModel",),
    )
)
def test_keyerror_raised(feature_descriptions, invalid_feature):
    fd = FeatureDescriptor(feature_descriptions)
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
def test_no_feature_mapping(feature_descriptions, nm_feature):
    fd = FeatureDescriptor(feature_descriptions)
    assert fd.mapping(nm_feature) is None


@pytest.mark.parametrize(
    ("nc_feature",),
    (
            ("Sex",),
            ("Height",),
            ("Date",),
    )
)
def test_no_feature_category(feature_descriptions, nc_feature):
    fd = FeatureDescriptor(feature_descriptions)
    assert fd.category(nc_feature) is None
