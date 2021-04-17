import pytest
from ml_dashboard.coordinator import Coordinator


@pytest.mark.parametrize(
    ("input_label",),
    (
            (0,),
            (1,),
    )
)
def test_coordinator_assert_classification_pos_label(data_classification_balanced, input_label):
    """Testing if assessing provided classification_pos_label returns the provided label if its in y values."""
    y = data_classification_balanced[1]
    Coordinator.y = y
    pos_label = 1
    pos_label = Coordinator._check_classification_pos_label(Coordinator, input_label)
    assert pos_label == input_label


@pytest.mark.parametrize(
    ("error_label",),
    (
            ("3",),
            ("Vegetables",),
            (4,),
    )
)
def test_coordinator_assert_classification_pos_label_error(data_classification_balanced, error_label):
    """Testing if coordinator raises an error when classification_pos_label is explicitly provided but it
    doesn't exist in y."""
    y = data_classification_balanced[1]
    Coordinator.y = y
    with pytest.raises(ValueError) as excinfo:
        c = Coordinator._check_classification_pos_label(Coordinator, error_label)

    assert str(error_label) in str(excinfo.value)


@pytest.mark.parametrize(
    ("warning_label",),
    (
            ("Fruits",),
            ("Dairy",),
            ("Sweets",),
    )
)
def test_coordinator_assert_classification_pos_label_warning(data_multiclass, root_path_to_package, warning_label):
    """Testing if warning is raised when classification_pos_label is explicitly provided for multiclass y."""
    y = data_multiclass[1]
    Coordinator.y = y
    Coordinator._force_classification_pos_label_multiclass_flag = False
    pos_label = 1
    with pytest.warns(Warning) as warninfo:
        pos_label = Coordinator._check_classification_pos_label(Coordinator, warning_label)
    assert "classification_pos_label will be ignored" in warninfo[0].message.args[0]
    assert pos_label is None


@pytest.mark.parametrize(
    ("input_label",),
    (
            ("Fruits",),
            ("Dairy",),
            ("Sweets",),
    )
)
def test_coordinator_assert_classification_pos_label_forced(data_multiclass, root_path_to_package, input_label):
    """Testing if classification_pos_label is set correctly for multiclass y when flag for forcing it is set to True."""
    y = data_multiclass[1]
    Coordinator.y = y
    Coordinator._force_classification_pos_label_multiclass_flag = True
    pos_label = 1
    pos_label = Coordinator._check_classification_pos_label(Coordinator, input_label)
    assert pos_label == input_label