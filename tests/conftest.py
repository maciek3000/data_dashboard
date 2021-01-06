import pytest
import random
import json
import tempfile, os, io

import pandas as pd
import numpy as np
from scipy.stats import truncnorm, skewnorm

from create_model.descriptor import FeatureDescriptor


@pytest.fixture
def feature_descriptions():
    d = "description"
    m = "mapping"
    descriptions = {
        "Sex": {
            d: "Sex of the Participant"
        },
        "AgeGroup": {
            d: "Age Group of the Participant.",
            m: {
                "18": "Between 18 and 22",
                "23": "Between 23 and 27",
                "28": "Between 28 and 32",
                "33": "Between 33 and 37",
                "38": "Between 38 and 42",
                "43": "Between 43 and 47",
                "48": "Between 48 and 52",
                "53": "Between 53 and 57",
                "58": "Between 58 and 62"
            }
        },
        "Height": {
            d: "Height of the Participant"
        },
        "Date": {
            d: "Date of the Transaction"
        },
        "Product": {
            d: "Product bought within the Transaction"
        },
        "Price": {
            d: "Price of the Product"
        },
        "bool": {
            d: "Random Flag"
        },
        "Target": {
            d: "Was the Transaction satisfactory?\nTarget Feature",
            m: {
                "1": "Yes",
                "0": "No"
            }
        }
    }

    return descriptions

@pytest.fixture
def temp_json_file(feature_descriptions):

    fd, path = tempfile.mkstemp()
    with open(path, "w") as tmp:
        json.dump(feature_descriptions, tmp)
        tmp.flush()

    with open(path, "r") as f:
        yield f

    os.close(fd)
    os.unlink(path)

@pytest.fixture
def feature_descriptor(temp_json_file):
    fd = FeatureDescriptor(temp_json_file)
    return fd

@pytest.fixture
def test_data_classification_balanced():

    random_seed = 56

    columns = ["Sex", "AgeGroup", "Height", "Date", "Product", "Price", "bool", "Target"]
    length = 100

    random.seed(random_seed)
    np.random.seed(seed=random_seed)

    # 50/50 gender randomness
    sex_data = random.choices(["Male", "Female"], k=length)

    # 9 different Age groups, more younger people
    # Age is treated here as a Categorical Variable, because there are only 9 unique values presented
    age_data = random.choices(
        range(18, 63, 5),
        weights=([0.12]*5) + ([0.1]*4),
        k=length
    )

    # normal distribution of Height between 150 and 210 cm, sd = 5cm
    # https://stackoverflow.com/a/44308018
    def trunc_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd
        )

    height_data = trunc_normal(mean=180, sd=5, low=150, upp=210).rvs(length)

    # Dates between 01-Jan-2020 and 31-Dec-2020
    date_data = random.choices(pd.date_range(start="01/01/2020", end="31/12/2020"), k=length)

    # 10 Products to choose from
    product_list = ["Apples", "Bread", "Cheese", "Eggs", "Ketchup", "Honey", "Butter", "Bananas", "Oranges", "Cookies"]
    product_data = random.choices(product_list, k=length)

    # Price
    price_data = map(lambda x: abs(x) * 50, random.choices(skewnorm(1).rvs(length), k=length))

    # bool column
    bool_data = random.choices([True, False], k=length)

    # Target
    target_data = random.choices([1, 0], k=length)

    data = {
        col: series for col, series in zip(columns,
        [sex_data, age_data, height_data, date_data, product_data, price_data, bool_data, target_data])
    }
    df = pd.DataFrame(data=data)

    # random missing data
    np_rows = random.choices(range(length), k=10)
    np_cols = random.choices(range(len(df.columns) - 1), k=10)

    for row, col in zip(np_rows, np_cols):
        df.iloc[row, col] = np.nan

    X = df[columns[:-1]]
    y = df[columns[-1]]

    return X, y

@pytest.fixture
def expected_mapping():
    expected_mapping = {
        "Product": {
            "Apples": 0,
            "Bananas": 1,
            "Bread": 2,
            "Butter": 3,
            "Cheese": 4,
            "Cookies": 5,
            "Eggs": 6,
            "Honey": 7,
            "Ketchup": 8,
            "Oranges": 9
        },
        "Sex": {
            "Female": 0,
            "Male": 1
        },
        "AgeGroup": {
            18: 0,
            23: 1,
            28: 2,
            33: 3,
            38: 4,
            43: 5,
            48: 6,
            53: 7,
            58: 8
        },
        "bool": {
            0: 0,
            1: 1
        },
        "Target": {
            0: 0,
            1: 1
        }
    }
    return expected_mapping

@pytest.fixture
def test_html_table():
    _ = """
        <table>
        <thead><tr><th></th><th></th></tr></thead>
        <tbody>
        <tr><th>Sex</th><td></td></tr>
        <tr><th>Target</th><td></td></tr>
        <tr><th>Invalid Column</th><td></td></tr>
        </tbody>
        </table>
    """
    return _
