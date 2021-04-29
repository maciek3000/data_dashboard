import re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


def sanitize_input(input_str_list):
    r"""Replace all instances of 'weird' characters and spaces in string elements of a input_str_list sequence with '_'.

    Examples:
        [a/?, b\\t, c] --> [a__, b_, c]

    Args:
        input_str_list (list/iterable): list of strings

    Returns:
        list: list of newly created strings
    """
    p = r'["/\\.,?!:;\n\t\r\a\f\v\b \{\}\[\]\(\)]'
    new_input = [re.sub(p, "_", x) for x in input_str_list]
    return new_input


def sanitize_keys_in_dict(dictionary):
    """Sanitize keys in a dictionary so they are JS friendly.

    All keys in a dictionary are converted to strings and they are sanitized from any 'weird' characters present.

    Args:
        dictionary (dict): dictionary

    Returns:
        dict: new dict with sanitized keys
    """
    old_keys = list(dictionary.keys())
    new_keys = sanitize_input([str(key) for key in old_keys])
    new_dict = {}
    for old_key, new_key in zip(old_keys, new_keys):
        new_dict[new_key] = dictionary[old_key]

    return new_dict


def calculate_numerical_bins(series):
    """Calculate how many bins in a histogram should there be for a provided numerical series.

    Method used to calculate it is 'Freedman-Diaconis rule' -
    https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule . If the calculated result doesn't make sense (e.g. is
    higher than the number of rows or is 0 or lower), then return either number of rows from series or 1, respectively.

    Args:
        series (pandas.Series): numerical Series on which calculations happen

    Returns:
        int: number of bins
    """
    n = series.size
    iqr = series.quantile(0.75) - series.quantile(0.25)
    bins = (series.max() - series.min()) / ((2*iqr)*pow(n, (-1/3)))

    # if the number of bins is greater than n, n is returned
    # this can be the case for small, skewed series
    if bins > n or np.isnan(bins):
        bins = n

    # if the number of bins is 0 or lower, then 1 is returned
    if bins <= 0:
        bins = 1

    bins = int(round(bins, 0))
    return bins


def modify_histogram_edges(edges, interval_percentage=0.005):
    """Modify edges calculated by np.histogram by adding little space between them.

    numpy.histogram method computes bin_edges of a histogram and a default way to use them is to define
    left_edges = edges[:-1]; right_edges = edges[1:] and provide that to BarPlot Visualization. However, that method
    does not provide any space between one Bar and another, what leads to cluttering the Visualization and
    indistinguishable Bars. Moving right_edges to the left by a defined interval_percentage adds aforementioned space
    and makes Visualization look better and be more useful.

    Args:
        edges ({numpy.ndarray, list}): edges calculated by numpy.histogram method
        interval_percentage (float, optional): percentage by which right_edges of the histogram will be moved to
            the left, defaults to 0.005

    Returns:
        tuple: 2 element tuple of left_edges, right_edges values
    """
    # Adding space between the edges to visualize the data properly
    interval = (max(edges) - min(edges)) * interval_percentage  # 0.5%
    left_edges = edges[:-1]
    right_edges = [edge - interval for edge in edges[1:]]
    return left_edges, right_edges


def sort_strings(list_of_strings):
    """Return sorted list of strings in an ascending order by treating every string as if it would start with an
    uppercase letter.

    Examples:
        [toast2, Test3, Toast, test33] --> [Test3, test33, Toast, toast2]

    Args:
        list_of_strings (list): list of string elements

    Returns:
        list: sorted list of strings
    """
    return sorted(list_of_strings, key=lambda x: x.upper())


def reverse_sorting_order(str_name):
    """Return False if str_name ends with one of the err_strings suffixes. Otherwise, return True.

    Negation was introduced as the function is used to determine the order of the sorting depending on scoring
    function name: if scoring ends with "_error" or "_loss", it means that lower score is better. If it doesn't,
    then it means that higher score is better. As default sorting is ascending, reverse=True needs to be explicitly
    provided for the sorted function so that sortable object (e.g. list) is sorted in a descending fashion.

    Args:
        str_name (str): string to be evaluated

    Returns:
        bool: False if str_name ends with one of the strings defined inside the function; True otherwise.
    """
    err_strings = ("_error", "_loss")
    return not str_name.endswith(err_strings)


def obj_name(obj):
    """Return __name__ property of obj and if it's not defined, return it from obj's Parent Class.

    Args:
        obj (object): object

    Returns:
        str: __name__ of the object or it's Class
    """
    try:
        obj_str = obj.__name__
    except AttributeError:
        obj_str = type(obj).__name__
    return obj_str


def append_description(text, parsed_html):
    r"""Encompass in <span> HTML tag and append provided text to the end of provided parsed_html and return the
    created tag.

    Additionally, every \\n in text is replaced with <br> HTML tag.

    Args:
        text (str): text string
        parsed_html (bs4.BeautifulSoup): HTML text parsed with BeautifulSoup4

    Returns:
        bs4.Tag: Newly created <span> Tag.
    """
    new_tag = parsed_html.new_tag("span")
    lines = text.split("\n")
    new_tag.string = lines[0]
    if len(lines) > 1:
        for line in lines[1:]:  # [1:] because the first line is already appended before the loop
            new_tag.append(parsed_html.new_tag("br"))
            new_tag.append(parsed_html.new_string("{}".format(line)))

    return new_tag


def series_to_dict(series):
    r"""Return dictionary constructed from pandas.Series where keys are indexes of Series and values are respective
    string values from the array, but every instance of ', ' string is replaced with \\n.

    Args:
        series (pandas.Series): Series

    Returns:
        dict: dictionary with changed strings in values
    """
    dictionary = series.to_dict()
    dict_str = {key: str(item_str)[1:-1].replace(", ", "\n") for key, item_str in dictionary.items()}
    return dict_str


def replace_duplicate_str(duplicate_list_of_str):
    """Replace every duplicate instance of string element in duplicate_list_of_str with new string indicating which
    duplicate element it is. Return unchanged duplicate_list_of_str if there are no duplicates.

    Examples:
        [a, b, a, b, c, a] --> [a #1, b #1, a #2, b #2, c, a #3] \n
        [a, b] --> [a, b]

    Args:
        duplicate_list_of_str ({list, iterable}): list which can contain duplicate string elements

    Returns:
        list: list with counted duplicates or duplicate_list_of_str if no duplicates are present.
    """
    if len(set(duplicate_list_of_str)) != len(duplicate_list_of_str):
        base_cnt = Counter(duplicate_list_of_str)
        item_cnt = defaultdict(int)
        new_sequence = []

        for item in duplicate_list_of_str:
            if base_cnt[item] > 1:
                new_item = item + " #" + str(item_cnt[item] + 1)
                item_cnt[item] += 1
                new_sequence.append(new_item)
            else:
                new_sequence.append(item)

        return new_sequence
    else:
        return duplicate_list_of_str


def assess_models_names(model_tuples):
    """Replace first element of a tuple in a sequence of tuples with its appropriate obj_name and additionally check
    if the name is duplicated across other first elements of tuples and replace if it's the case.

    Args:
        model_tuples ({list, sequence}): sequence of 2-item tuples (object, value)

    Returns:
        list: list of new tuples with their first element appropriately replaced
    """
    models = [obj_name(tp[0]) for tp in model_tuples]
    new_names = replace_duplicate_str(models)
    new_tuple = [(new_name, value[1]) for new_name, value in zip(new_names, model_tuples)]
    return new_tuple


def make_pandas_data(data, desired_pandas_class):
    """Transform provided data object into a desired pandas class instance.

    data can be provided as either scipy.csr_matrix, numpy.ndarray, instance of desired_pandas_class or any other object
    that can be fed directly into a construction call of desired_pandas_class
    (e.g. dict in pd.DataFrame(data=your_dict)).

    desired_pandas_class should be pandas.DataFrame, pandas.Series or any other pandas object that would be creatable
    from either csr_matrix, ndarray or other allowed data container.

    Note:
        when data is an instance of desired_pandas_obj (DataFrame or Series), returned DataFrame or Series will
        have it's index reset.

    Args:
        data ({scipy.csr_matrix, numpy.ndarray, desired_pandas_class, object}): scipy.csr_matrix, numpy.ndarray,
            instance of desired_pandas_class or any other object that can be used in construction of
            desired_pandas_class object (desired_pandas_class(data))
        desired_pandas_class ({pandas.Series, pandas.DataFrame, object}): class from pandas module to contain data,
            e.g. pandas.DataFrame or pandas.Series

    Returns:
        desired_pandas_class: desired_pandas_class object with data inside.

    Raises:
        Exception: when data is not csr_matrix, ndarray or desired_pandas_class instance and it can't be used
            as an argument to create desired_pandas_class

    """
    if isinstance(data, csr_matrix):
        X_arr = data.toarray()
        if desired_pandas_class == pd.Series:  # special case as pd.Series needs 1d array
            X_arr = X_arr[0]
        new_data = desired_pandas_class(X_arr)
    elif isinstance(data, np.ndarray):
        new_data = desired_pandas_class(data)
    elif isinstance(data, desired_pandas_class):
        # resetting indexes: no use for them as of now and simple count will give the possibility to match rows
        # between raw and transformed data
        new_data = data.reset_index(drop=True)
    else:
        try:
            new_data = desired_pandas_class(data)
        except Exception:
            raise

    return new_data
