import os
import pytest
from bokeh.models.layouts import Tabs
from bokeh.layouts import Row

from ml_dashboard.views import ModelsViewClassification, ModelsViewRegression, ModelsViewMulticlass

@pytest.mark.parametrize(
    ("output_directory", "filename"),
    (
            ("C:\\path\\test", "test.js",),
            ("directory", "test_file.css",),
            ("D:/home/root/project", "init.html",)
    )
)
def test_output_path_to_file(output, output_directory, filename):
    """Testing if creating filepaths with provided output_directory works correctly."""
    output.output_directory = output_directory
    actual = output._path_to_file(filename)
    expected = os.path.join(output_directory, filename)

    assert actual == expected


@pytest.mark.parametrize(
    ("filename", "template"),
    (
            ("test.html", "<h1>Hello World</h1>"),
            ("test2.html", "This is test"),
            ("file.txt", "This is another test"),
    )
)
def test_output_write_html(output, filename, template, tmpdir):
    """Testing if writing content to the file works correctly."""
    output._write_html(filename, template)

    created_file = os.path.join(tmpdir, filename)

    assert os.path.exists(created_file)
    with open(created_file) as f:
        assert f.read() == template


@pytest.mark.parametrize(
    ("problem_type", "expected_result"),
    (
            ("classification", ModelsViewClassification),
            ("regression", ModelsViewRegression),
            ("multiclass", ModelsViewMulticlass)
    )
)
def test_models_view_creator(output, problem_type, expected_result):
    """Testing if output creates a correct ModelsView based on a provided problem type."""
    if problem_type == "classification":
        problem = output.model_finder._classification
    elif problem_type == "regression":
        problem = output.model_finder._regression
    else:
        problem = output.model_finder._multiclass

    actual_result = output._models_view_creator(problem)
    assert isinstance(actual_result, expected_result)


@pytest.mark.parametrize(
    ("incorrect_problem_type",),
    (
            ("class",),
            ("reg",),
            (None,),
            (True,),
            (False,),
            (10,),
    )
)
def test_models_view_creator_error(output, incorrect_problem_type):
    """Testing if _models_view_creator raises an Exception when an incorrect problem type is provided."""
    with pytest.raises(ValueError) as excinfo:
        _ = output._models_view_creator(incorrect_problem_type)
    assert str(incorrect_problem_type) in str(excinfo.value)


@pytest.mark.parametrize(
    ("problem_type", "expected_result"),
    (
            ("classification", (Tabs, list)),
            ("regression", (Tabs, Tabs)),
            ("multiclass", (None, Row))
    )
)
def test_models_plot_output(
        output, model_finder_classification_fitted, model_finder_regression_fitted, model_finder_multiclass_fitted,
        problem_type, expected_result
):
    """Testing if output creates output of a correct type based on a provided problem type."""
    if problem_type == "classification":
        output.model_finder = model_finder_classification_fitted
        problem = output.model_finder._classification
    elif problem_type == "regression":
        output.model_finder = model_finder_regression_fitted
        problem = output.model_finder._regression
    else:
        output.model_finder = model_finder_multiclass_fitted
        problem = output.model_finder._multiclass

    actual_result = output._models_plot_output(problem)

    try:
        assert isinstance(actual_result[0], expected_result[0])
    except TypeError:
        assert actual_result[0] is None and expected_result[0] is None

    try:
        assert isinstance(actual_result[1], expected_result[1])
    except TypeError:
        assert actual_result[1] is None and expected_result[1] is None


@pytest.mark.parametrize(
    ("incorrect_problem_type",),
    (
            ("class",),
            ("reg",),
            (None,),
            (True,),
            (False,),
            (10,),
    )
)
def test_models_plot_output_error(output, incorrect_problem_type):
    """Testing if _models_view_creator raises an Exception when an incorrect problem type is provided."""
    with pytest.raises(ValueError) as excinfo:
        _ = output._models_plot_output(incorrect_problem_type)
    assert str(incorrect_problem_type) in str(excinfo.value)
