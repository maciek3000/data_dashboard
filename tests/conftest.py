import pytest
import random

import pandas as pd
from scipy.stats import truncnorm, skewnorm


@pytest.fixture
def test_data_classification_balanced():

    random_seed = 56

    columns = ["Sex", "Age", "Height", "Date", "Product", "Price", "Target"]
    length = 100

    random.seed(random_seed)

    # 50/50 gender randomness
    sex_data = random.choices(["Male", "Female"], k=length)

    # 9 different Age groups, more younger people
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
    price_data = random.choices(skewnorm(1).rvs(length), k=length)

    # Target
    target_data = random.choices([1, 0], k=length)

    data = {
        col: series for col, series in zip(columns, [sex_data, age_data, height_data, date_data, product_data, price_data, target_data])
    }
    df = pd.DataFrame(data=data)

    X = df[columns[:-1]]
    y = df[columns[-1]]

    return X, y
