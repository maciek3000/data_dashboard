from ml_dashboard.output import Output
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
