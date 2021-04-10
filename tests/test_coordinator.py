import pytest
from ml_dashboard.coordinator import Coordinator

@pytest.mark.parametrize(
    ("error_label",),
    (
            ("3",),
            ("Vegetables",),
            (4,),
    )
)
def test_coordinator_assert_classification_pos_label_error(data_classification_balanced, seed, error_label):
    """Testing if coordinator raises an error when classification_pos_label is explicitly provided but it
    doesn't exist in y."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    with pytest.raises(ValueError) as excinfo:
        c = Coordinator(X, y, "test_directory", classification_pos_label=error_label)

    assert str(error_label) in str(excinfo.value)


@pytest.mark.parametrize(
    ("warning_label",),
    (
            ("Fruits",),
            ("Dairy",),
            ("Sweets",),
    )
)
def test_coordinator_assert_classification_pos_label_warning(data_multiclass, seed, root_path_to_package, warning_label):
    """Testing if warning is raised when classification_pos_label is explicitly provided for multiclass y."""
    y = data_multiclass[1]
    Coordinator.y = y
    pos_label = 1
    with pytest.warns(Warning) as warninfo:
        pos_label = Coordinator._check_classification_pos_label(Coordinator, warning_label)
    assert "classification_pos_label will be ignored" in warninfo[0].message.args[0]
    assert pos_label is None