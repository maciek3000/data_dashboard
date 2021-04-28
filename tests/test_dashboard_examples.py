import pytest
from data_dashboard.examples.examples import iris, boston, diabetes, digits, wine, breast_cancer
from data_dashboard.dashboard import Dashboard


@pytest.mark.parametrize(
    ("input_data",),
    (
            (iris,),
            (boston,),
            (diabetes,),
            (digits,),
            (wine,),
            (breast_cancer,),
    )
)
def test_dashboard_working_examples(input_data, tmpdir):
    """Testing if creating a default Dashboard with included examples works (no Exceptions are raised)."""
    X, y, descriptions = input_data()
    dsh = Dashboard(X, y, tmpdir, descriptions)
    dsh.create_dashboard()

    assert True
