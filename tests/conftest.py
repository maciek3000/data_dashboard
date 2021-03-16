# built-in
import pytest
import random
import json
import tempfile
import os
import copy

# libraries
import pandas as pd
import numpy as np
from scipy.stats import truncnorm, skewnorm

# this package
from create_model.descriptor import FeatureDescriptor
from create_model.features import Features


@pytest.fixture
def feature_descriptions():
    d = FeatureDescriptor._description
    m = FeatureDescriptor._mapping
    c = FeatureDescriptor._category

    descriptions = {
        "Sex": {
            d: "Sex of the Participant"
        },
        "AgeGroup": {
            # d: "Age Group of the Participant.",
            m: {
                18: "Between 18 and 22",
                23: "Between 23 and 27",
                28: "Between 28 and 32",
                33: "Between 33 and 37",
                38: "Between 38 and 42",
                43: "Between 43 and 47",
                48: "Between 48 and 52",
                53: "Between 53 and 57",
                58: "Between 58 and 62"
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
                1: "Yes",
                0: "No"
            },
            c: "cat"
        }
    }

    return descriptions


@pytest.fixture
def feature_descriptor(feature_descriptions):
    fd = FeatureDescriptor(feature_descriptions)
    return fd


@pytest.fixture
def feature_descriptor_forced_categories(feature_descriptions):
    c = FeatureDescriptor._category

    cat_cols = ["Height", "Price"]
    num_cols = ["bool", "AgeGroup", "Target"]

    new_descriptions = copy.deepcopy(feature_descriptions)

    for cat in cat_cols:
        new_descriptions[cat][c] = "cat"

    for cat in num_cols:
        new_descriptions[cat][c] = "num"

    fd = FeatureDescriptor(new_descriptions)

    return fd


@pytest.fixture
def feature_descriptor_broken(feature_descriptions):
    broken_features = ["Target", "AgeGroup"]
    for feat in broken_features:
        internal = feature_descriptions[feat]
        _ = {}
        for key, item in internal.items():
            _[str(key)] = item
        feature_descriptions[feat] = _

    fd = FeatureDescriptor(feature_descriptions)
    return fd


@pytest.fixture
def data_classification_balanced():

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
        if col != df.columns.to_list().index("bool"):
            df.iloc[row, col] = np.nan

    X = df[columns[:-1]]
    y = df[columns[-1]]

    return X, y


@pytest.fixture
def categorical_features():
    return ["AgeGroup", "bool", "Product", "Sex", "Target"]


@pytest.fixture
def numerical_features():
    return ["Height", "Price"]


@pytest.fixture
def fixture_features(data_classification_balanced, feature_descriptor):
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)
    return f


@pytest.fixture
def expected_raw_mapping():
    expected_raw_mapping = {
        "Product": {
            "Apples": 1,
            "Bananas": 2,
            "Bread": 3,
            "Butter": 4,
            "Cheese": 5,
            "Cookies": 6,
            "Eggs": 7,
            "Honey": 8,
            "Ketchup": 9,
            "Oranges": 10
        },
        "Sex": {
            "Female": 1,
            "Male": 2
        },
        "AgeGroup": {
            18: 1,
            23: 2,
            28: 3,
            33: 4,
            38: 5,
            43: 6,
            48: 7,
            53: 8,
            58: 9
        },
        "bool": {
            0: 1,
            1: 2
        },
        "Target": {
            0: 1,
            1: 2
        }
    }
    return expected_raw_mapping

@pytest.fixture
def expected_mapping():
    expected_mapping = {
        "Product": {
            1: "Apples",
            2: "Bananas",
            3: "Bread",
            4: "Butter",
            5: "Cheese",
            6: "Cookies",
            7: "Eggs",
            8: "Honey",
            9: "Ketchup",
            10: "Oranges"
        },
        "Sex": {
            1: "Female",
            2: "Male"
        },
        "AgeGroup": {
            1: "Between 18 and 22",
            2: "Between 23 and 27",
            3: "Between 28 and 32",
            4: "Between 33 and 37",
            5: "Between 38 and 42",
            6: "Between 43 and 47",
            7: "Between 48 and 52",
            8: "Between 53 and 57",
            9: "Between 58 and 62"
        },
        "bool": {
            1: 0,
            2: 1
        },
        "Target": {
            1: "No",
            2: "Yes"
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
