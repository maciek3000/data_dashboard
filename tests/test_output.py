import pytest
import os
import datetime
import pandas as pd
from bokeh.models.layouts import Tabs
from bokeh.layouts import Row
from data_dashboard.views import ModelsViewClassification, ModelsViewRegression, ModelsViewMulticlass


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


def test_output_static_path(output, tmpdir):
    """Testing if static directory is created in the provided output_directory."""
    assert output.static_path() == os.path.join(tmpdir, "static")


def test_output_assets_path(output, tmpdir):
    """Testing if assets directory is created in the provided output_directory."""
    assert output.assets_path() == os.path.join(tmpdir, "assets")


def test_output_logs_path(output, tmpdir):
    """Testing if logs directory is created in the provided output_directory."""
    assert output.logs_path() == os.path.join(tmpdir, "logs")


@pytest.mark.parametrize(
    ("input_time", "expected_directory_name"),
    (
            (datetime.datetime(2020, 12, 1, 15, 34, 59), "01122020153459"),
            (datetime.datetime(1990, 1, 13, 23, 59, 00), "13011990235900"),
            (datetime.datetime(2056, 9, 23, 1, 1, 34), "23092056010134")
    )
)
def test_output_create_logs_directory(output, tmpdir, input_time, expected_directory_name):
    """Testing if subdirectory in logs is created correctly and with a correct name based on a provided time."""
    expected_result = os.path.join(tmpdir, "logs", expected_directory_name)
    actual_result = output._create_logs_directory(input_time)
    assert actual_result == expected_result
    assert os.path.isdir(expected_result)


def test_output_write_logs_files_created(output, tmpdir):
    """Testing if writing log csv log files works correctly."""
    test_date = datetime.datetime(2020, 3, 1, 14, 0, 34)
    log_dir = "01032020140034"
    filenames = [output._search_results_csv, output._quicksearch_results_csv, output._gridsearch_results_csv]
    expected_filepaths = [os.path.join(tmpdir, "logs", log_dir, filename) for filename in filenames]

    for f in expected_filepaths:
        assert not os.path.exists(f)

    output._write_logs(test_date)

    for f in expected_filepaths:
        assert os.path.exists(f)


def test_output_write_logs_csv_content(output, tmpdir, model_finder_classification_fitted):
    """Testing if csv files written as logs are the same as those in model_finder properties."""
    test_date = datetime.datetime(2020, 3, 1, 14, 0, 34)
    log_dir = "01032020140034"
    filenames = [output._search_results_csv, output._quicksearch_results_csv, output._gridsearch_results_csv]
    expected_filepaths = [os.path.join(tmpdir, "logs", log_dir, filename) for filename in filenames]
    mf = model_finder_classification_fitted
    expected_dfs = [mf.search_results(None), mf.quicksearch_results(), mf.gridsearch_results()]

    output._write_logs(test_date)

    for f, df in zip(expected_filepaths, expected_dfs):
        actual_df = pd.read_csv(f, index_col=0)
        expected_df = df
        assert actual_df.shape == expected_df.shape  # checking only shape because of round differences/nans


def test_output_write_logs_one_df_missing(output, tmpdir, model_finder_classification_fitted):
    """Testing that csv files are not created when appropriate result df is None."""
    test_date = datetime.datetime(2020, 3, 1, 14, 0, 34)
    log_dir = "01032020140034"
    filenames = [output._search_results_csv, output._gridsearch_results_csv]
    expected_filepaths = [os.path.join(tmpdir, "logs", log_dir, filename) for filename in filenames]
    mf = model_finder_classification_fitted
    mf._quicksearch_results = None

    output._write_logs(test_date)

    assert len([n for n in os.listdir(os.path.join(tmpdir, "logs", log_dir))]) == len(filenames)
    for f in expected_filepaths:
        assert os.path.exists(f)


@pytest.mark.parametrize(
    ("input_directory",),
    (
            ("static",),
            (os.path.join("static", "static2"),),
            (os.path.join("a", "b", "c"),)
    )
)
def test_output_create_output_directory(output, tmpdir, input_directory):
    """Testing if creating output_directory works in case it doesn't exist."""
    directory = os.path.join(tmpdir, input_directory)
    output.output_directory = directory
    output._create_output_directory()
    assert os.path.isdir(directory)


@pytest.mark.parametrize(
    ("input_directory",),
    (
            ("static",),
            (os.path.join("a", "b"),)
    )
)
def test_dashboard_output_directory_exists(output, tmpdir, input_directory):
    """Testing if create_output_directory does not interfere when the directory already exists."""
    directory = os.path.join(tmpdir, input_directory)
    os.makedirs(directory)
    assert os.path.isdir(directory)
    output.output_directory = directory
    output._create_output_directory()
    assert os.path.isdir(directory)


def test_output_create_subdirectories(output, tmpdir):
    """Testing if static and assets subdirectories are created correctly."""
    directories = ["static", "assets"]
    expected_directories = [os.path.join(tmpdir, d) for d in directories]

    for d in expected_directories:
        assert not os.path.isdir(d)

    output._create_subdirectories()

    for d in expected_directories:
        assert os.path.isdir(d)


def test_output_copy_static(output, tmpdir, root_path_to_package):
    """Testing if static files are copied correctly to the output_directory folder."""
    directory, pkg_name = root_path_to_package[0], root_path_to_package[1]
    base_files = [os.path.join(directory, pkg_name, "static", f) for f in output._static_files_names]
    expected_filepaths = [os.path.join(tmpdir, "static", f) for f in output._static_files_names]

    output._create_subdirectories()
    output._copy_static()

    for actual_file, expected_file in zip(expected_filepaths, base_files):
        assert os.path.exists(actual_file)
        assert open(actual_file, "r").readlines() == open(expected_file, "r").readlines()


def test_output_overview_path(output, tmpdir):
    """Testing if overview HTML file path is created correctly."""
    expected_path = os.path.join(tmpdir, "overview.html")
    actual_path = output.overview_file()
    assert actual_path == expected_path


def test_output_features_path(output, tmpdir):
    """Testing if features HTML file path is created correctly."""
    expected_path = os.path.join(tmpdir, "features.html")
    actual_path = output.features_file()
    assert actual_path == expected_path


def test_output_models_path(output, tmpdir):
    """Testing if models HTML file path is created correctly."""
    expected_path = os.path.join(tmpdir, "models.html")
    actual_path = output.models_file()
    assert actual_path == expected_path