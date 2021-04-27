import pytest
from data_dashboard.descriptor import FeatureDescriptor


def test_feature_get(feature_descriptions):
    """Testing if FeatureDescriptor[feature] syntax works."""
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
    """Testing if mapping() function returns proper values."""
    mapping = FeatureDescriptor._mapping
    fd = FeatureDescriptor(feature_descriptions)
    assert fd.mapping(feature) == feature_descriptions[feature][mapping]


def test_feature_category(feature_descriptions):
    """Testing if category() function returns proper values."""
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
    """Testing if KeyError is raised when incorrect feature name is provided."""
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
    """Testing if None is returned when the feature does not have corresponding mapping defined."""
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
    """Testing if None is returned when the feature does not have corresponding category defined."""
    fd = FeatureDescriptor(feature_descriptions)
    assert fd.category(nc_feature) is None
