import pytest
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge, PassiveAggressiveClassifier, SGDRegressor, SGDClassifier, LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, log_loss

from data_dashboard.functions import append_description, assess_models_names, calculate_numerical_bins, make_pandas_data
from data_dashboard.functions import modify_histogram_edges, obj_name, replace_duplicate_str, reverse_sorting_order
from data_dashboard.functions import sanitize_input, series_to_dict, sort_strings


@pytest.mark.parametrize(
    ("header_index", "description", "expected_html"),
    (
            (0, "Sex of the Participant", "<span>Sex of the Participant</span>"),
            (1, "Was the Transaction satisfactory?\nTarget Feature", "<span>Was the Transaction "
                                                                     "satisfactory?<br/>Target Feature</span>"),
            (2, "Price of the Product", "<span>Price of the Product</span>"),
            (3, "Description not\n Available", "<span>Description not<br/> Available</span>"),
            (4, "Height of the Participant", "<span>Height of the Participant</span>"),
            (5, "Product bought within the Transaction", "<span>Product bought within the Transaction</span>"),
            (6, "Random Flag", "<span>Random Flag</span>")
    )
)
def test_append_descriptions(html_test_table, header_index, description, expected_html):
    """Testing if appending description (wrapped in HTML tags) to the element works properly."""
    html_table = BeautifulSoup(html_test_table, "html.parser")
    actual_html = str(append_description(description, html_table))

    assert actual_html == expected_html


@pytest.mark.parametrize(
    ("input_tuple_list", "expected_names"),
    (
            ([(SGDRegressor, ("1", "2")), (SGDRegressor, ("3", "4"), SGDClassifier, "test")],
             ["SGDRegressor #1", "SGDRegressor #2", "SGDClassifier"]),
            (
                    [
                        (SGDRegressor, "test"),
                        (SGDClassifier, "test2"),
                        (Lasso, "test3"),
                        (SGDClassifier, "test4"),
                        (SGDRegressor, "test4"),
                        (LinearRegression, "test5"),
                        (SGDClassifier, "test6")
                    ],
                    ["SGDRegressor #1", "SGDClassifier #1", "Lasso", "SGDClassifier #2",
                     "SGDRegressor #2", "LinearRegression", "SGDClassifier #3"]
            )

    )
)
def test_assess_model_names(input_tuple_list, expected_names):
    """Testing if replacing duplicate model names in a tuple of (model, values) works correctly."""
    expected_results = []
    for name, tp in zip(expected_names, input_tuple_list):
        expected_results.append((name, tp[1]))

    actual_results = assess_models_names(input_tuple_list)
    assert actual_results == expected_results


@pytest.mark.parametrize(
    ("input_series", "expected_result"),
    (
            ([1, 5, 5, 6, 7, 9, 10, 10, 10, 11, 15, 17, 20], 4),  # 0.25 - 6; 0.75 - 11
            ([1, 1, 1, 1, 2, 5, 70, 90, 5, 5, 4, 3, 2, 1, 8, 8], 16),  # 0.25 - 1; 0.75 - 5.75
            ([6, 3, 4, 3, 0, 9, 5, 3, 3, 3, 8, 1, 0, 8, 0, 4, 0, 2, 9, 3, 7, 1, 0, 2, 3, 6, 9, 1, 0, 6, 1, 6, 5, 3, 4,
              3, 1, 0, 1, 2, 3, 8, 9, 1, 1, 0, 6, 2, 3, 8, 1, 1, 1, 1, 6], 3),  # 0.25 - 1; 0.75 - 6
            ([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9], 6),  # 0.25 - 4; 0.75 - 6
    )
)
def test_calculate_numerical_bins(input_series, expected_result):
    """Testing if calculate_numerical_bins() correctly calculates the number of bins."""
    srs = pd.Series(input_series)
    actual_result = calculate_numerical_bins(srs)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_data", "expected_pandas_obj", "expected_result"),
    (
            (
                csr_matrix([0, 1, 0]),
                pd.DataFrame,
                pd.DataFrame(data={0: [0], 1: [1], 2: [0]})
            ),
            (
                np.array([[1, 2], [3, 4]]),
                pd.DataFrame,
                pd.DataFrame(data={0: [1, 3], 1: [2, 4]})
            ),
            (
                pd.DataFrame(data={"a": [1, 2, 3], "b": [3, 4, None]}, index=["a", "b", "c"]),
                pd.DataFrame,
                pd.DataFrame(data={"a": [1, 2, 3], "b": [3, 4, None]}, index=[0, 1, 2])
            ),
            (
                {"a": [1, 2, 3], "b": [2, 3, 4]},
                pd.DataFrame,
                pd.DataFrame(data={"a": [1, 2, 3], "b": [2, 3, 4]})
            ),
            (
                np.array([4, 5, 6]),
                pd.Series,
                pd.Series([4, 5, 6], dtype="int32")
            ),
            (
                pd.Series([1, 2, 3], index=["a", "b", "c"]),
                pd.Series,
                pd.Series([1, 2, 3], index=[0, 1, 2])
            )
    )
)
def test_make_pandas_data(input_data, expected_pandas_obj, expected_result):
    """Testing if make_pandas_data returns correct output when different input data is provided."""
    actual_result = make_pandas_data(input_data, expected_pandas_obj)
    assert str(actual_result) == str(expected_result)


@pytest.mark.parametrize(
    ("wrong_input",),
    (
            ("wrong-string",),
            (True,),
            (str,),
    )
)
def test_make_pandas_data_error(wrong_input):
    """Testing if make_pandas raises an error when incorrect input is provided."""
    with pytest.raises(Exception):
        make_pandas_data(wrong_input, pd.DataFrame)

@pytest.mark.parametrize(
    ("input_edges", "interval_percentage", "expected_right_edge"),
    (
            ([5, 10, 15, 20, 25, 30], 0.005, [9.875, 14.875, 19.875, 24.875, 29.875]),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.1, [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.005, [1.955, 2.955, 3.955, 4.955, 5.955, 6.955, 7.955, 8.955, 9.955])
    )
)
def test_modify_histogram_edges(input_edges, interval_percentage, expected_right_edge):
    """Testing if modify_histogram_edges() correctly returns arrays for left and right edges."""
    expected_left_edge = input_edges[:-1]

    actual_left_edge, actual_right_edge = modify_histogram_edges(input_edges, interval_percentage)

    assert actual_left_edge == expected_left_edge
    assert actual_right_edge == expected_right_edge


@pytest.mark.parametrize(
    ("obj", "expected_result"),
    (
            (dict(), "dict"),
            (dict, "dict"),
            (None, "NoneType"),
            (lambda x: x, "<lambda>"),
            (Ridge, "Ridge"),
            (Ridge(), "Ridge"),
            (PassiveAggressiveClassifier, "PassiveAggressiveClassifier"),
            (PassiveAggressiveClassifier(), "PassiveAggressiveClassifier"),
            (roc_auc_score, "roc_auc_score"),
            (mean_squared_error, "mean_squared_error"),
            (r2_score, "r2_score"),
            (log_loss, "log_loss")

    )
)
def test_obj_name(obj, expected_result):
    """Testing if returned string representation of object from obj_name() function is correct."""
    actual_result = obj_name(obj)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_list", "expected_result"),
    (
            (["test", "not-test", "test", "another-test"], ["test #1", "not-test", "test #2", "another-test"]),
            (["a", "a", "A", "b", "b", "c", "c", "c"], ["a #1", "a #2", "A", "b #1", "b #2", "c #1", "c #2", "c #3"]),
            (["a", "a", "a", "a"], ["a #1", "a #2", "a #3", "a #4"]),
            (["a", "b", "c", "d"], ["a", "b", "c", "d"]),
            (["a", "b", "b", "b", "c"], ["a", "b #1", "b #2", "b #3", "c"])
    )
)
def test_replace_duplicate_str(input_list, expected_result):
    """Testing if replacing duplicate entries in a list works correctly."""
    actual_result = replace_duplicate_str(input_list)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_str", "expected_result"),
    (
            ("roc_auc_score", True),
            ("mean_squared_error", False),
            ("mean_negative_loss", False),
            ("loss_score", True),
            ("error", True),
            ("loss", True),
            ("error_loss", False),
            ("loss_error", False),
            ("test_string", True),
            ("qualityloss", True)
    )
)
def test_reverse_sorting_order(input_str, expected_result):
    """Testing if assessment of sorting order from reverse_sorting_order() is correct."""
    assert reverse_sorting_order(input_str) == expected_result


@pytest.mark.parametrize(
    ("input_list", "expected_result"),
    (
            (["test\n", "test\b", "aa\aaa"], ["test_", "test_", "aa_aa"]),
            (["tes?t", "ano/ther t\fres\v\t", "target..."], ["tes_t", "ano_ther t_res__", "target___"]),
            (["\\hey", "!test", ":hello:", "hel;o"], ["_hey", "_test", "_hello_", "hel_o"])
    )
)
def test_sanitize_input(input_list, expected_result):
    """Testing if sanitizing input list (replacing invalid characters) works properly."""
    actual_result = sanitize_input(input_list)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("param_series", "expected_result"),
    (
            (pd.Series({"1": {'C': 1.0, 'alpha': 0.001}, "2": {'max_depth': 10.0, 'criterion': 'gini'}}),
             {"1": "'C': 1.0\n'alpha': 0.001", "2": "'max_depth': 10.0\n'criterion': 'gini'"}),
            (pd.Series({
                "test1": {"C": 1.0, "alpha": 0.001, "depth": "10"},
                "test2": {"aa": 1.0, "bb": "cc", "test": "test"},
                "test3": {"test1": 10.0, "C": 0.001, "alpha": 1000.0}
            }),
             {
                 "test1": "'C': 1.0\n'alpha': 0.001\n'depth': '10'",
                 "test2": "'aa': 1.0\n'bb': 'cc'\n'test': 'test'",
                 "test3": "'test1': 10.0\n'C': 0.001\n'alpha': 1000.0"
             })
    )
)
def test_series_to_dict(param_series, expected_result):
    actual_result = series_to_dict(param_series)
    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("input_string", "expected_output"),
    (
            (["string1", "String2", "STRING3"], ["string1", "String2", "STRING3"]),
            (["bool", "Bool", "abcd"], ["abcd", "bool", "Bool"]),
            (["zzz", "ZZA", "bbb", "bbc", "Aaa", "aab"], ["Aaa", "aab", "bbb", "bbc", "ZZA", "zzz"])
    )
)
def test_sort_strings(input_string, expected_output):
    """Testing if sort_strings sorting works correctly."""
    assert sorted(input_string) != expected_output
    actual_output = sort_strings(input_string)
    assert actual_output == expected_output
