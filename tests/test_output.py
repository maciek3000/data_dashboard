from create_model.output import Output
import os
import pytest


@pytest.mark.parametrize(
    ("output_directory", "filename"),
    (
            ("C:\\path\\test", "test.js",),
            ("directory", "test_file.css",),
            ("D:/home/root/project", "init.html",)
    )
)
def test_output_path_to_file(analyzer_fixture, root_path_to_package, output_directory, filename):
    """Testing if creating filepaths with provided output_directory works correctly."""
    package_path = root_path_to_package[0]
    package_name = root_path_to_package[1]
    o = Output(package_path, analyzer_fixture, package_name)
    o.output_directory = output_directory
    actual = o._path_to_file(filename)
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
def test_output_write_html(analyzer_fixture, root_path_to_package, tmpdir, filename, template):
    """Testing if writing content to the file works correctly."""
    package_path = root_path_to_package[0]
    package_name = root_path_to_package[1]
    o = Output(package_path, analyzer_fixture, package_name)
    o.output_directory = tmpdir
    o._write_html(filename, template)

    created_file = os.path.join(tmpdir, filename)

    assert os.path.exists(created_file)
    with open(created_file) as f:
        assert f.read() == template
