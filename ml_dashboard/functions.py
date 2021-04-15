from collections import Counter, defaultdict
import numpy as np
import re
from scipy.sparse import csr_matrix
import pandas as pd


def sanitize_input(input_str_list):
    p = r'["/\\.,?!:;\n\t\r\a\f\v\b]'
    new_input = [re.sub(p, "_", x) for x in input_str_list]
    return new_input


def calculate_numerical_bins(series):

    # https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule
    n = series.size
    iqr = series.quantile(0.75) - series.quantile(0.25)
    bins = (series.max() - series.min()) / ((2*iqr)*pow(n, (-1/3)))

    # if the number of bins is greater than n, n is returned
    # this can be the case for small, skewed series
    if bins > n or np.isnan(bins):
        bins = n

    if bins <= 0:
        bins = 1

    bins = int(round(bins, 0))
    return bins


def modify_histogram_edges(edges, interval_percentage=0.005):
    # Adding space between the edges to visualize the data properly
    interval = (max(edges) - min(edges)) * interval_percentage  # 0.5%
    left_edges = edges[:-1]
    right_edges = [edge - interval for edge in edges[1:]]
    return left_edges, right_edges


def sort_strings(list_of_strings):
    return sorted(list_of_strings, key=lambda x: x.upper())


def reverse_sorting_order(str_name):
    """If str_name ends with err_strings defined in functions, returns False. Otherwise, returns True.

        Negation was introduced as the function is used to determine the order of the sorting depending on scoring
        function name: if scoring ends with "_error" or "_loss", it means that lower score is better. If it doesn't,
        then it means that higher score is better. As default sorting is ascending, reverse=True needs to be explicitly
        provided for the object (e.g. list) to be sorted in a descending fashion.
    """
    # functions ending with _error or _loss return a value to minimize, the lower the better.
    err_strings = ("_error", "_loss")
    # boolean output will get fed to "reversed" argument of sorted function: True -> descending; False -> ascending
    # if str ends with one of those, then it means that lower is better -> ascending sort.
    return not str_name.endswith(err_strings)


def obj_name(obj):
    """Checks if obj defines __name__ property and if not, gets it from it's Parent Class."""
    try:
        obj_str = obj.__name__
    except AttributeError:
        obj_str = type(obj).__name__
    return obj_str


def info_symbol_html():
    # TODO: check if used anywhere
    return "<span class='info-symbol'>&#x1F6C8;</span>"


def append_description(description, parsed_html):
    # adding <span> that will hold description of a feature
    # every \n is replaced with <br> tag
    new_tag = parsed_html.new_tag("span")
    lines = description.split("\n")
    new_tag.string = lines[0]
    if len(lines) > 1:
        for line in lines[1:]:
            new_tag.append(parsed_html.new_tag("br"))
            new_tag.append(parsed_html.new_string("{}".format(line)))
    return new_tag


def series_to_dict(srs):
    params = srs.to_dict()
    dict_str = {model: str(params_str)[1:-1].replace(", ", "\n") for model, params_str in params.items()}
    return dict_str


def replace_duplicate_str(duplicate_list_of_str):

    if len(set(duplicate_list_of_str)) != len(duplicate_list_of_str):
        base_cnt = Counter(duplicate_list_of_str)
        item_cnt = defaultdict(int)
        new_container = []

        for item in duplicate_list_of_str:
            if base_cnt[item] > 1:
                new_item = item + " #" + str(item_cnt[item] + 1)
                item_cnt[item] += 1
                new_container.append(new_item)
            else:
                new_container.append(item)

        return new_container
    else:
        return duplicate_list_of_str


def assess_models_names(model_tuple):
    models = [obj_name(tp[0]) for tp in model_tuple]
    new_names = replace_duplicate_str(models)
    new_tuple = [(new_name, value[1]) for new_name, value in zip(new_names, model_tuple)]
    return new_tuple


def make_pandas_data(data, desired_pandas_obj):
    if isinstance(data, csr_matrix):
        X_arr = data.toarray()
        if desired_pandas_obj == pd.Series:
            X_arr = X_arr[0]
        new_data = desired_pandas_obj(X_arr)
    elif isinstance(data, np.ndarray):
        new_data = desired_pandas_obj(data)
    elif isinstance(data, desired_pandas_obj):
        # resetting indexes: no use for them as of now and simple count will give the possibility to match rows
        # between raw and transformed data
        # copy: to make sure that changes won't affect the original data
        new_data = data.copy().reset_index(drop=True)
    else:
        try:
            new_data = desired_pandas_obj(data)
        except:
            raise

    return new_data