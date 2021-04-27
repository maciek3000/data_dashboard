# built-in
import pytest
import random
import os
import copy

# libraries
import pandas as pd
import numpy as np
from jinja2 import Template
from scipy.stats import truncnorm, skewnorm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import f1_score, precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# this package
from data_dashboard.descriptor import FeatureDescriptor
from data_dashboard.features import Features
from data_dashboard.analyzer import Analyzer
from data_dashboard.transformer import Transformer
from data_dashboard.model_finder import ModelFinder
from data_dashboard.output import Output
from data_dashboard.dashboard import Dashboard


@pytest.fixture
def feature_descriptions():
    """Descriptions for test data."""
    d = FeatureDescriptor._description
    m = FeatureDescriptor._mapping
    c = FeatureDescriptor._category

    descriptions = {
        "Sex": {
            d: "Sex of the Participant"
        },
        "AgeGroup": {
            # d: "Age Group of the Participant.",  # description removed for testing purposes
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
    """Fixture FeatureDescriptor."""
    fd = FeatureDescriptor(feature_descriptions)
    return fd


@pytest.fixture
def feature_descriptor_forced_categories(feature_descriptions):
    """Fixture FeatureDescriptor with forced categories."""
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
    """Fixture FeatureDescriptor with str keys instead of int."""
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
    """Test data with 'balanced' ratio of 0s and 1s in 'Target' variable."""
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
        weights=([0.12] * 5) + ([0.1] * 4),
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

    data_list = [sex_data, age_data, height_data, date_data, product_data, price_data, bool_data, target_data]
    data = {
        col: series for col, series in zip(columns, data_list)
    }

    df = pd.DataFrame(data=data)

    # random missing data
    np_rows = random.choices(range(length), k=10)
    np_cols = random.choices(range(len(df.columns) - 1), k=10)

    # not including nans in bool column as it automatically converts the category to Object
    for row, col in zip(np_rows, np_cols):
        if col != df.columns.to_list().index("bool"):
            df.iloc[row, col] = np.nan

    X = df[columns[:-1]]
    y = df[columns[-1]]

    return X, y


@pytest.fixture
def categorical_features():
    """Categorical Features names in data_classification_balanced."""
    return ["AgeGroup", "bool", "Product", "Sex", "Target"]


@pytest.fixture
def numerical_features():
    """Numerical Features names in data_classification_balanced."""
    return ["Height", "Price"]


@pytest.fixture
def fixture_features(data_classification_balanced, feature_descriptor):
    """Fixture Features object for data_classification_balanced test data."""
    X, y = data_classification_balanced
    f = Features(X, y, feature_descriptor)
    return f


@pytest.fixture
def expected_raw_mapping():
    """Expected 'raw' mapping of values in data_classification_balanced test data."""
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
            False: 1,
            True: 2
        },
        "Target": {
            0: 1,
            1: 2
        }
    }
    return expected_raw_mapping


@pytest.fixture
def expected_mapping():
    """Expected final mapping (from FeaturesDescriptor) for data_classification_balanced test data."""
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
            1: False,
            2: True
        },
        "Target": {
            1: "No",
            2: "Yes"
        }
    }
    return expected_mapping


@pytest.fixture
def html_test_table():
    """Test table for data_classification_balanced test_data."""
    _ = """
        <table>
        <thead><tr><th></th><th></th></tr></thead>
        <tbody>
        <tr><th>Sex</th><td></td></tr>
        <tr><th>Target</th><td></td></tr>
        <tr><th>Price</th><td></td></tr>
        <tr><th>AgeGroup</th><td></td></tr>
        <tr><th>Height</th><td></td></tr>
        <tr><th>Product</th><td></td></tr>
        <tr><th>bool</th><td></td></tr>
        </tbody>
        </table>
    """
    return _


@pytest.fixture
def analyzer_fixture(fixture_features):
    """Fixture Analyzer for data_classification_balanced test data."""
    return Analyzer(fixture_features)


@pytest.fixture
def root_path_to_package():
    """Path to modules (package) and modules (package) name."""
    package_name = "data_dashboard"
    root_path = os.path.split(os.getcwd())[0]

    return root_path, package_name


@pytest.fixture
def seed():
    "Fixture random seed."
    return 1010


@pytest.fixture
def data_regression(data_classification_balanced):
    """Test data with Price feature as a target variable."""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    target = "Price"
    feats = df.columns.to_list()
    feats.remove(target)
    X = df[feats]
    median = df[target].describe()["50%"]
    y = df[target].fillna(median)
    return X, y


@pytest.fixture
def data_multiclass(data_classification_balanced):
    """Test data with multiclass 'Product Type' target variable."""
    X = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    y = pd.Series(np.where(
        X["Product"].isin(["Apples", "Oranges", "Bananas"]), "Fruits",
        np.where(
            X["Product"].isin(["Honey", "Cookies"]), "Sweets",
            "Dairy")
    ), name="Product Type")
    y = y.fillna("Fruits")
    return X, y


@pytest.fixture
def preprocessor_X(categorical_features, numerical_features, seed):
    """Base Fixture preprocessor X."""
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        QuantileTransformer(output_distribution="normal", random_state=seed),
        StandardScaler()
    )

    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    )

    col_transformer = ColumnTransformer(
        transformers=[
            ("numerical", numeric_transformer, numerical_features),
            ("categorical", categorical_transformer, categorical_features)
        ]
    )
    return col_transformer


@pytest.fixture
def transformer_classification(categorical_features, numerical_features, seed, preprocessor_X):
    """Transformer for data_classification_balanced test data."""
    categorical_features.remove("Target")
    tr = Transformer(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_type="Categorical",
        random_state=seed
    )

    tr.preprocessor_X = preprocessor_X
    tr.preprocessor_y = LabelEncoder()

    return tr


@pytest.fixture
def transformer_regression(categorical_features, numerical_features, seed, preprocessor_X):
    """Transformer for data_regression test data."""
    numerical_features.remove("Price")
    tr = Transformer(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_type="Numerical",
        random_state=seed
    )

    tr.preprocessor_X = preprocessor_X
    tr.preprocessor_y = FunctionTransformer(lambda x: x)

    return tr


@pytest.fixture
def transformer_multiclass(categorical_features, numerical_features, seed, preprocessor_X):
    """Transformer for data_multiclass test data."""
    tr = Transformer(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_type="Categorical",
        random_state=seed
    )

    tr.preprocessor_X = preprocessor_X
    tr.preprocessor_y = LabelEncoder()

    return tr


@pytest.fixture
def transformer_classification_fitted(transformer_classification, data_classification_balanced):
    """Fitted Transformer for data_classification_balanced test data."""
    transformer_classification.fit(data_classification_balanced[0])
    transformer_classification.fit_y(data_classification_balanced[1])
    return transformer_classification


@pytest.fixture
def transformed_classification_data(data_classification_balanced, transformer_classification):
    """Transformed data_classification_balanced test data."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    X = X.drop(["Date"], axis=1)
    transformer_classification.fit(X)
    transformer_classification.fit_y(y)
    return transformer_classification.transform(X), transformer_classification.transform_y(y)


@pytest.fixture
def transformed_regression_data(data_regression, transformer_regression):
    """Transformed data_regression test data."""
    X = data_regression[0]
    y = data_regression[1]
    X = X.drop(["Date"], axis=1)
    transformer_regression.fit(X)
    transformer_regression.fit_y(y)
    return transformer_regression.transform(X), transformer_regression.transform_y(y)


@pytest.fixture
def transformed_multiclass_data(data_multiclass, transformer_multiclass):
    """Transformed data_multiclass test data."""
    X = data_multiclass[0]
    y = data_multiclass[1]
    X = X.drop(["Date"], axis=1)
    transformer_multiclass.fit(X)
    transformer_multiclass.fit_y(y)
    return transformer_multiclass.transform(X), transformer_multiclass.transform_y(y)


@pytest.fixture
def split_dataset_classification(data_classification_balanced, transformer_classification, seed):
    """Train/test split of data_classification_balanced test data."""
    X = data_classification_balanced[0]
    y = data_classification_balanced[1]
    X = X.drop(["Date"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    t = transformer_classification
    t.fit(X_train)
    t.fit_y(y_train)
    return t.transform(X_train), t.transform(X_test), t.transform_y(y_train), t.transform_y(y_test)


@pytest.fixture
def split_dataset_numerical(data_regression, transformer_regression, seed):
    """Train/test split of data_regression test data."""
    X = data_regression[0]
    y = data_regression[1]
    X = X.drop(["Date"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    t = transformer_regression
    t.fit(X_train)
    t.fit_y(y_train)
    return t.transform(X_train), t.transform(X_test), t.transform_y(y_train), t.transform_y(y_test)


@pytest.fixture
def split_dataset_multiclass(data_multiclass, transformer_multiclass, seed):
    """Train/test split of data_multiclass test data."""
    X = data_multiclass[0]
    y = data_multiclass[1]
    X = X.drop(["Date"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    t = transformer_multiclass
    t.fit(X_train)
    t.fit_y(y_train)
    return t.transform(X_train), t.transform(X_test), t.transform_y(y_train), t.transform_y(y_test)


@pytest.fixture
def chosen_classifiers_grid():
    """Test 'Model': parameters pairs for classification."""
    _ = {
        LogisticRegression: {
            "tol": np.logspace(-1, 0, 2),
            "C": np.logspace(-1, 0, 2)
        },
        DecisionTreeClassifier: {
            "max_depth": [10, None],
            "criterion": ["gini", "entropy"],
        },
        SVC: {
            "tol": np.logspace(-1, 0, 2),
            "C": np.logspace(-1, 0, 2)
        }
    }

    return _


@pytest.fixture
def chosen_regressors_grid():
    """Test 'Model': parameters pairs for regression."""
    _ = {
        Ridge: {
            "alpha": np.logspace(-7, -4, 4),
        },
        DecisionTreeRegressor: {
            "max_depth": [10, None],
            "criterion": ["mae", "mse"]
        },
        SVR: {
            "tol": np.logspace(-1, 0, 2),
            "C": np.logspace(-2, -1, 2)
        }
    }

    return _


@pytest.fixture
def multiclass_scorings():
    """Wrapped scoring functions for multiclass problem."""
    scorings = [
        (f1_score, {"average": "weighted"}, "f1_score_weighted"),
        (precision_score, {"average": "weighted"}, "precision_score_weighted")
    ]

    new_scorings = []
    for scoring, params, fname in scorings:
        def f():
            f_sc = scoring
            f_par = params
            f_nam = fname

            def make_scoring(y_true, y_score):
                make_scoring.__name__ = f_nam
                return f_sc(y_true, y_score, **f_par)
            return make_scoring

        new_scorings.append(f())

    return new_scorings


@pytest.fixture
def model_finder_classification(
        transformed_classification_data, split_dataset_classification, chosen_classifiers_grid, seed
):
    """Fixture ModelFinder for classification problem."""
    X = transformed_classification_data[0]
    y = transformed_classification_data[1]
    X_train, X_test, y_train, y_test = split_dataset_classification
    mf = ModelFinder(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_type="categorical",
        random_state=seed
    )

    mf.default_models = chosen_classifiers_grid
    return mf


@pytest.fixture
def model_finder_regression(transformed_regression_data, split_dataset_numerical, chosen_regressors_grid, seed):
    """Fixture ModelFinder for regression problem."""
    X = transformed_regression_data[0]
    y = transformed_regression_data[1]
    X_train, X_test, y_train, y_test = split_dataset_numerical
    mf = ModelFinder(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_type="numerical",
        random_state=seed
    )

    mf.default_models = chosen_regressors_grid
    return mf


@pytest.fixture
def model_finder_multiclass(
        transformed_multiclass_data, split_dataset_multiclass, chosen_classifiers_grid, multiclass_scorings, seed
):
    """Fixture ModelFinder for multiclass problem."""
    X = transformed_multiclass_data[0]
    y = transformed_multiclass_data[1]
    X_train, X_test, y_train, y_test = split_dataset_multiclass
    mf = ModelFinder(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_type="categorical",
        random_state=seed
    )

    mf.default_models = chosen_classifiers_grid
    mf.scoring_functions = multiclass_scorings
    return mf


@pytest.fixture
def model_finder_classification_fitted(model_finder_classification):
    """Fixture fitted ModelFinder for classification problem."""
    model_finder_classification.search_and_fit(mode="quick")
    return model_finder_classification


@pytest.fixture
def model_finder_regression_fitted(model_finder_regression):
    """Fixture fitted ModelFinder for regression problem."""
    model_finder_regression.search_and_fit(mode="quick")
    return model_finder_regression


@pytest.fixture
def model_finder_multiclass_fitted(model_finder_multiclass):
    """Fixture fitted ModelFinder for multiclass problem."""
    model_finder_multiclass.search_and_fit(mode="quick")
    return model_finder_multiclass


@pytest.fixture
def output(
        analyzer_fixture, transformer_classification, fixture_features, model_finder_classification_fitted,
        tmpdir, root_path_to_package, data_classification_balanced, split_dataset_classification, seed
):
    """Fixture Output object."""
    X, y = data_classification_balanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test = split_dataset_classification

    o = Output(
        output_directory=tmpdir,
        package_name=root_path_to_package[1],
        pre_transformed_columns=[],
        features=fixture_features,
        analyzer=analyzer_fixture,
        transformer=transformer_classification,
        model_finder=model_finder_classification_fitted,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        transformed_X_train=transformed_X_train,
        transformed_X_test=transformed_X_test,
        transformed_y_train=transformed_y_train,
        transformed_y_test=transformed_y_test
    )
    return o


@pytest.fixture
def dashboard(data_classification_balanced, tmpdir, seed):
    """Fixture Dashboard object."""
    X, y = data_classification_balanced
    d = Dashboard(X, y, tmpdir, random_state=seed)
    return d


@pytest.fixture
def template():
    """Fixture test template."""
    return Template("Test template")
