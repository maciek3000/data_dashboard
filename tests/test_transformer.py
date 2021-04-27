import pytest
import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.preprocessing import LabelEncoder, FunctionTransformer, OneHotEncoder, QuantileTransformer, StandardScaler
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from data_dashboard.transformer import Transformer
from data_dashboard.transformer import WrapperFunctionTransformer


@pytest.mark.parametrize(
    ("test_func",),
    (
            (lambda x: x,),
            (lambda x: 1,),
            (lambda x: x**2,),
    )
)
def test_wrapper_func_transformer(test_func):
    """Testing if WrapperFunctionTransformer still has functionality of an underlying FunctionTransformer."""
    test_arr = np.array([1, 1, 1, 2, 3, 4, 5]).reshape(-1, 1)

    tr = FunctionTransformer(func=test_func)
    wrap_tr = WrapperFunctionTransformer("test", clone(tr))

    expected_arr = tr.fit_transform(test_arr)
    actual_arr = wrap_tr.fit_transform(test_arr)

    assert np.array_equal(actual_arr, expected_arr)
    assert str(wrap_tr) != str(tr)


@pytest.mark.parametrize(
    ("test_text",),
    (
            ("Test1",),
            ("Test2",),
            ("",),
    )

)
def test_wrapper_func_transformer_str(test_text):
    """Testing if str() function of WrapperFunctionTransformer returns text provided as an argument."""
    wrap_tr = WrapperFunctionTransformer(test_text, FunctionTransformer())
    assert str(wrap_tr) == test_text


def test_transformer_create_preprocessor_X(categorical_features, numerical_features):
    """Testing if X preprocessor correctly assigns steps to columns depending on their type."""
    categorical_features.remove("Target")
    tr = Transformer(categorical_features, numerical_features, "Categorical")
    preprocessor = tr._create_preprocessor_X()
    expected_steps = [("numerical", numerical_features), ("categorical", categorical_features)]

    actual_steps = [(item[0], item[2]) for item in preprocessor.transformers]

    for step in expected_steps:
        assert step in actual_steps

    assert len(actual_steps) == len(expected_steps)


@pytest.mark.parametrize(
    ("target_type", "expected_function"),
    (
            ("Categorical", LabelEncoder()),
            ("Numerical", WrapperFunctionTransformer("test", FunctionTransformer(lambda x: x)))
    )
)
def test_transformer_create_preprocessor_y(categorical_features, numerical_features, target_type, expected_function):
    """Testing if y preprocessor is created correctly."""
    tr = Transformer(categorical_features, numerical_features, target_type)
    preprocessor = tr._create_default_transformer_y()

    assert type(preprocessor).__name__ == type(expected_function).__name__


@pytest.mark.parametrize(
    ("transformed_feature",),
    (
            ("Price",),
            ("bool",),
            ("AgeGroup",)
    )
)
def test_transformer_preprocessor_X_remainder(
        categorical_features, numerical_features, data_classification_balanced, expected_raw_mapping,
        transformed_feature
):
    """Testing if feature not declared in either categorical or numerical features passes through unchanged."""
    categorical_features.remove("Target")
    categorical_features = [f for f in categorical_features if f != transformed_feature]
    numerical_features = [f for f in numerical_features if f != transformed_feature]
    X = data_classification_balanced[0].drop(["Date"], axis=1)

    if transformed_feature in expected_raw_mapping.keys():
        X[transformed_feature] = X[transformed_feature].replace(expected_raw_mapping[transformed_feature])

    tr = Transformer(categorical_features, numerical_features, "Numerical")
    tr.fit(X)
    transformed = tr.transform(X)
    try:
        transformed = transformed.toarray()
    except AttributeError:
        transformed = transformed

    cols = tr.transformed_columns() + [transformed_feature]
    actual_result = pd.DataFrame(transformed, columns=cols)

    assert np.allclose(actual_result[transformed_feature].to_numpy(), X[transformed_feature].to_numpy(), equal_nan=True)
    # checking if there is only one column with transformed_feature (no derivations)
    assert sum([1 for col in cols if transformed_feature in col]) == 1


@pytest.mark.parametrize(
    ("transformed_features",),
    (
            (["Height", "Price"],),
            (["Price", "AgeGroup"],),
    )
)
def test_transformer_preprocessor_X_remainder_order(
        categorical_features, numerical_features, data_classification_balanced, expected_raw_mapping,
        transformed_features
):
    """Testing if remainder portion of ColumnTransformer returns the columns in the expected (alphabetical) order."""
    categorical_features.remove("Target")
    categorical_features = [f for f in categorical_features if f not in transformed_features]
    numerical_features = [f for f in numerical_features if f not in transformed_features]

    X = data_classification_balanced[0].drop(["Date"], axis=1)

    tr = Transformer(categorical_features, numerical_features, "Numerical")
    tr.fit(X)
    transformed = tr.transform(X)
    try:
        transformed = transformed.toarray()
    except AttributeError:
        transformed = transformed

    cols = tr.transformed_columns() + sorted(transformed_features)
    actual_result = pd.DataFrame(transformed, columns=cols)

    for col in transformed_features:
        assert np.allclose(actual_result[col].to_numpy(), X[col].to_numpy(), equal_nan=True)


@pytest.mark.parametrize(
    ("target_type",),
    (
            (None,),
            ("Target",),
            ("categorical",),
            (10,),
            (True,),
            (np.nan,),
    )
)
def test_transformer_create_preprocessor_y_invalid_target_type(categorical_features, numerical_features, target_type):
    """Testing if ._create_preprocessor_y raises an Exception when invalid target_type is provided"""
    tr = Transformer(categorical_features, numerical_features, "Categorical")  # initiating with proper type
    tr.target_type = target_type
    with pytest.raises(ValueError) as excinfo:
        preprocessor = tr._create_default_transformer_y()
    assert "should be Categorical or Numerical" in str(excinfo.value)


@pytest.mark.parametrize(
    ("feature_name",),
    (
            ("AgeGroup",),
            ("bool",),
            ("Product",),
            ("Sex",),
            ("Target",),
    )
)
def test_transformer_transform_y_categorical(
        data_classification_balanced, categorical_features, numerical_features, expected_raw_mapping, feature_name
):
    """Testing if fit_y() and transform_y() are changing provided y correctly (when y is categorical)"""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)

    target = df[feature_name]
    mapping = {key: int(item - 1) for key, item in expected_raw_mapping[feature_name].items()}
    mapping[np.nan] = max(mapping.values()) + 1
    expected_result = target.replace(mapping).array

    tr = Transformer(categorical_features, numerical_features, "Categorical")
    actual_result = tr.fit_transform_y(target)

    assert np.array_equal(actual_result, expected_result)


@pytest.mark.parametrize(
    ("feature_name",),
    (
            ("Height",),
            ("Price",),
    )
)
def test_transformer_transform_y_numerical(
        data_classification_balanced, categorical_features, numerical_features, feature_name
):
    """Testing if fit_y() and transform_y() are changing provided y correctly (when y is numerical)"""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    target = df[feature_name]
    expected_result = target.array

    tr = Transformer(categorical_features, numerical_features, "Numerical")
    actual_result = tr.fit_transform_y(target)

    assert np.allclose(actual_result, expected_result, equal_nan=True)


@pytest.mark.parametrize(
    ("feature", "classification_pos_label",),
    (
            ("Sex", "Female"),
            ("Sex", "Male"),
            ("Target", 0),
            ("Product", "Apples"),
            ("Product", "Potato")
    )
)
def test_transformer_transform_y_classification_pos_label(
        data_classification_balanced, categorical_features, numerical_features, feature, classification_pos_label,
):
    """Testing if transformer correctly changes mappings of y when explicit classification_pos_label is provided."""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    expected_result = df[feature].apply(lambda x: 1 if x == classification_pos_label else 0)
    tr = Transformer(
        categorical_features, numerical_features, "Categorical", classification_pos_label=classification_pos_label
    )
    actual_result = tr.fit_transform_y(df[feature])
    assert np.array_equal(actual_result, expected_result)


@pytest.mark.parametrize(
    ("classification_pos_label",),
    (
            ("Fruits",),
            ("Sweets",),
            ("Dairy",),
    )
)
def test_transformer_transform_y_classification_pos_label_multiclass(
        data_multiclass, categorical_features, numerical_features, classification_pos_label,
):
    """Testing if transformer correctly changes mappings of y when explicit classification_pos_label is provided
    for multiclass problem (so the mapping changes it to classification problem)."""
    y = data_multiclass[1]
    mapping = {
        "Fruits": 0,
        "Sweets": 0,
        "Dairy": 0,
        classification_pos_label: 1  # overwriting with test choice
    }
    expected_result = y.replace(mapping)
    tr = Transformer(
        categorical_features, numerical_features, "Categorical", classification_pos_label=classification_pos_label
    )
    actual_result = tr.fit_transform_y(y)
    assert np.array_equal(actual_result, expected_result)


@pytest.mark.parametrize(
    ("feature_name", "csr_matrix_flag"),
    (
            ("AgeGroup", True),
            ("bool", False),
            ("Product", True),
            ("Sex", False),
            ("Target", False),
    )
)
def test_transformer_transform_X_categorical(data_classification_balanced, feature_name, csr_matrix_flag):
    """Testing if every categorical column from a test data is transformed correctly."""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    # replacing for SimpleImputer which cant handle bool dtype
    df["bool"] = df["bool"].replace({False: 0, True: 1})
    feature = df[feature_name]
    most_frequent = feature.value_counts(dropna=False).index[0]
    feature = feature.fillna(most_frequent)
    expected_result = OneHotEncoder(handle_unknown="ignore").fit_transform(feature.to_numpy().reshape(-1, 1)).toarray()

    tr = Transformer([feature_name], [], "Categorical")
    actual_result = tr.fit_transform(pd.DataFrame(df[feature_name]))

    # for n > 2 unique values, output is a csr_matrix
    if csr_matrix_flag:
        actual_result = actual_result.toarray()

    assert pd.DataFrame(actual_result).equals(pd.DataFrame(expected_result))


@pytest.mark.parametrize(
    ("feature_name",),
    (
            ("Height",),
            ("Price",),
    )
)
def test_transformer_transform_X_numerical(data_classification_balanced, feature_name):
    """Testing if every numerical column from a test data is transformed correctly."""
    random_state = 1
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    feature = df[feature_name]
    median = feature.describe()["50%"]
    feature = feature.fillna(median).to_numpy().reshape(-1, 1)
    feature = QuantileTransformer(output_distribution="normal", random_state=random_state).fit_transform(
        feature)
    feature = StandardScaler().fit_transform(feature)
    expected_result = feature

    tr = Transformer([], [feature_name], "Categorical", random_state=random_state)
    actual_result = tr.fit_transform(pd.DataFrame(df[feature_name]))

    assert np.allclose(actual_result, expected_result)


@pytest.mark.parametrize(
    ("y_column", "expected_result"),
    (
            ("Sex", ["Female", "Male", np.nan]),
            ("AgeGroup", [18, 23, 28, 33, 38, 43, 48, 53, 58]),
            ("Product",
             ["Apples", "Bananas", "Bread", "Butter", "Cheese", "Cookies", "Eggs", "Honey", "Ketchup", "Oranges",
              np.nan]),
            ("Target", [0, 1])
    )
)
def test_transformer_y_classes_classification(
        data_classification_balanced, categorical_features, numerical_features, y_column, expected_result, seed
):
    """Testing if classification transformer returns correct classes from y."""
    df = pd.concat([data_classification_balanced[0], data_classification_balanced[1]], axis=1)
    cat_feats = categorical_features.remove(y_column)
    tr = Transformer(cat_feats, numerical_features, "Categorical", random_state=seed)
    tr.fit_y(df[y_column])

    actual_result = tr.y_classes().tolist()

    assert actual_result == expected_result


def test_transformer_y_classes_regression_error(categorical_features, numerical_features, seed):
    """Testing if y_classes raises an Error when target_type is provided as Numerical."""
    tr = Transformer(categorical_features, numerical_features, "Numerical", seed)
    with pytest.raises(ValueError):
        _ = tr.y_classes()


@pytest.mark.parametrize(
    ("feature", "category"),
    (
            ("Sex", "Categorical"),
            ("Product", "Categorical"),
            ("Height", "Numerical"),
            ("Price", "Numerical"),
    )
)
def test_transformer_transformers(transformer_classification_fitted, feature, category, seed):
    """Testing if correct transformers are returned when provided with a feature name."""
    if category == "Categorical":
        expected_result = ["SimpleImputer(strategy='most_frequent')", "OneHotEncoder(handle_unknown='ignore')"]
    elif category == "Numerical":
        expected_result = [
            "SimpleImputer(strategy='median')",
            "QuantileTransformer(output_distribution='normal', random_state={seed})".format(seed=seed),
            "StandardScaler()"
        ]
    else:
        raise

    actual_result = list(map(str, transformer_classification_fitted.transformers(feature)))

    assert actual_result == expected_result


def test_transformer_custom_transformers(categorical_features, numerical_features, seed):
    """Testing if setting custom transformers in Transformer object works correctly."""
    categorical_tr = [SimpleImputer(strategy="most_frequent"), OrdinalEncoder()]
    numerical_tr = [SimpleImputer(), PowerTransformer()]
    y_transformer = FunctionTransformer()

    tr = Transformer(categorical_features, numerical_features, "Categorical", seed)
    tr.set_custom_preprocessor_X(categorical_tr, numerical_tr)
    tr.set_custom_preprocessor_y(y_transformer)

    assert tr.categorical_transformers == categorical_tr
    assert tr.numerical_transformers == numerical_tr
    assert tr.y_transformer == y_transformer


@pytest.mark.parametrize(
    ("custom_transformers", "tr_type"),
    (
            ([SimpleImputer(strategy="constant", fill_value="Missing"), OneHotEncoder()], "categorical"),
            ([SimpleImputer(strategy="average"), PowerTransformer()], "numerical")
    )
)
def test_transformer_custom_transformer_none_transformers(
        categorical_features, numerical_features, seed, custom_transformers, tr_type
):
    """Testing if setting custom transformers with only one type of transformers provided works correctly."""
    tr = Transformer(categorical_features, numerical_features, "Categorical", seed)

    if tr_type == "categorical":
        tr.set_custom_preprocessor_X(categorical_transformers=custom_transformers)
        expected_numerical = tr._default_numerical_transformers
        expected_categorical = custom_transformers
    elif tr_type == "numerical":
        tr.set_custom_preprocessor_X(numerical_transformers=custom_transformers)
        expected_numerical = custom_transformers
        expected_categorical = tr._default_categorical_transformers
    else:
        raise

    assert tr.categorical_transformers == expected_categorical
    assert tr.numerical_transformers == expected_numerical


@pytest.mark.parametrize(
    ("feature",),
    (
            ("Date",),
            ("Unused Feature",),
    )
)
def test_transformer_transformers_untransformed_feature(transformer_classification_fitted, feature):
    """Testing if no transformers are returned from transformations() function when unused feature is provided."""
    expected_result = None
    actual_result = transformer_classification_fitted.transformers(feature)

    assert actual_result == expected_result


def test_transformer_transformed_columns_names(transformer_classification_fitted):
    """Testing if transformed_columns() returns correct combination of transformed column names."""
    categorical_columns = [
        "AgeGroup_18",
        "AgeGroup_23",
        "AgeGroup_28",
        "AgeGroup_33",
        "AgeGroup_38",
        "AgeGroup_43",
        "AgeGroup_48",
        "AgeGroup_53",
        "AgeGroup_58",
        "bool_False",
        "bool_True",
        "Product_Apples",
        "Product_Bananas",
        "Product_Bread",
        "Product_Butter",
        "Product_Cheese",
        "Product_Cookies",
        "Product_Eggs",
        "Product_Honey",
        "Product_Ketchup",
        "Product_Oranges",
        "Sex_Female",
        "Sex_Male"
    ]
    expected_result = ["Height", "Price"] + categorical_columns

    actual_result = transformer_classification_fitted.transformed_columns()

    assert actual_result == expected_result


@pytest.mark.parametrize(
    ("feature",),
    (
            ("Sex",),
            ("bool",),
            ("Product",),
            ("AgeGroup",),
    )
)
def test_transformer_transformed_columns_categorical_encoding(
        transformer_classification_fitted, data_classification_balanced, feature
):
    """Testing if transformed_columns() returns transformed column names in correct order (OneHotEncoding)."""
    categorical_columns = [
        "AgeGroup_18",
        "AgeGroup_23",
        "AgeGroup_28",
        "AgeGroup_33",
        "AgeGroup_38",
        "AgeGroup_43",
        "AgeGroup_48",
        "AgeGroup_53",
        "AgeGroup_58",
        "bool_False",
        "bool_True",
        "Product_Apples",
        "Product_Bananas",
        "Product_Bread",
        "Product_Butter",
        "Product_Cheese",
        "Product_Cookies",
        "Product_Eggs",
        "Product_Honey",
        "Product_Ketchup",
        "Product_Oranges",
        "Sex_Female",
        "Sex_Male"
    ]
    col_names = ["Height", "Price"] + categorical_columns
    df = data_classification_balanced[0]
    # replacing for SimpleImputer which cant handle bool dtype
    df["bool"] = df["bool"].replace({False: 0, True: 1})
    test_feature = df[feature]
    expected_result = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ).fit_transform(test_feature.to_numpy().reshape(-1, 1)).toarray()

    expected_col_names = [col for col in col_names if feature in col]
    df = pd.DataFrame(
        data=transformer_classification_fitted.transform(data_classification_balanced[0]).toarray(),
        columns=col_names
    )
    actual_result = df[expected_col_names].to_numpy()

    assert len(df.columns) == len(col_names)
    assert np.array_equal(actual_result, expected_result)


def test_transformer_transformed_columns_no_one_hot_encoder(
        transformer_classification_fitted, data_classification_balanced
):
    """Testing if columns from transformed_columns() are correctly calculated when there is no OneHotEncoder present
    in the preprocessor."""
    transformer_classification_fitted.set_custom_preprocessor_X(
        categorical_transformers=[SimpleImputer(strategy="most_frequent")],
        numerical_transformers=[SimpleImputer()]
    )
    transformer_classification_fitted.fit(data_classification_balanced[0])
    expected_results = ["Height", "Price", "AgeGroup", "bool", "Product", "Sex"]
    actual_results = transformer_classification_fitted.transformed_columns()

    assert actual_results == expected_results


@pytest.mark.parametrize(
    ("feature", "category", "expected_columns"),
    (
            ("AgeGroup", "Categorical", ["AgeGroup_18", "AgeGroup_23", "AgeGroup_28", "AgeGroup_33", "AgeGroup_38",
                                         "AgeGroup_43", "AgeGroup_48", "AgeGroup_53", "AgeGroup_58"]),
            ("bool", "Categorical", ["bool_False", "bool_True"]),
            ("Height", "Numerical", ["Height"]),
            ("Price", "Numerical", ["Price"]),
            ("Product", "Categorical", ["Product_Apples", "Product_Bananas", "Product_Bread", "Product_Butter",
                                        "Product_Cheese", "Product_Cookies", "Product_Eggs", "Product_Honey",
                                        "Product_Ketchup", "Product_Oranges"]),
            ("Sex", "Categorical", ["Sex_Female", "Sex_Male"])
    )
)
def test_transformer_transformations(transformer_classification_fitted, feature, category, expected_columns, seed):
    if category == "Categorical":
        expected_transformers = [SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")]
    elif category == "Numerical":
        expected_transformers = [
            SimpleImputer(strategy="median"),
            QuantileTransformer(output_distribution="normal", random_state=seed),
            StandardScaler()
        ]
    else:
        raise

    results = transformer_classification_fitted.transformations()
    actual_results = results[feature]

    assert len(results) == 6
    assert str(actual_results[0]) == str(expected_transformers)
    assert actual_results[1] == expected_columns


@pytest.mark.parametrize(
    ("feature",),
    (
            ("Height",),
            ("Price",),
    )
)
def test_transformer_normal_transformations(
        transformer_classification_fitted, data_classification_balanced, feature, seed
):
    """Testing if normal_transformations() method returns correct 'normal' transformations for a given feature."""
    expected_transformers = [QuantileTransformer, PowerTransformer]
    X = data_classification_balanced[0][feature]
    X_train, X_test = train_test_split(X, random_state=seed)
    actual_results = transformer_classification_fitted.normal_transformations(
        X_train.to_numpy().reshape(-1, 1),
        X_test.to_numpy().reshape(-1, 1)
    )

    assert len(actual_results) == 3
    for transformer, results in actual_results:
        assert transformer.__class__ in expected_transformers
        assert results.shape[0] == X_test.shape[0]
        assert not np.array_equal(results, X_test.to_numpy())


@pytest.mark.parametrize(
    ("input_train_array", "input_test_array", "expected_transformers_len"),
    (
            ([1, 1, 1, 1, 2, 3, 4], [2, 3, 4], 3),
            ([-1, 1, 2, 3, 4, 5], [2, 3, 4], 2),
            ([1, 1, 2, 3, 4, 5], [-2, 3, 4], 2),
            ([np.nan, 0.1, 1, 2, 3, 4], [2, 3, 4], 3),
            ([2, 2, 3, 4, 5, 6, 7], [np.nan, -3, 4], 2),
            ([0, 1, 22, 3, 5, 6], [1, 2, 3], 2),
            ([1, 1, 2, 3, 4, 6, 5], [0, 1, 2], 2)
    )
)
def test_transformer_normal_transformations_negative_input(
        transformer_classification_fitted, input_train_array, input_test_array, expected_transformers_len, seed
):
    """Testing if normal_transformations are returned correctly when input has negative values."""
    expected_transformers = {
        "QuantileTransformer(output_distribution='normal', random_state={seed})".format(seed=seed),
        "PowerTransformer()"  # yeo-johnson is default
    }
    X_train = np.array(input_train_array).reshape(-1, 1)
    X_test = np.array(input_test_array).reshape(-1, 1)
    actual_results = transformer_classification_fitted.normal_transformations(X_train, X_test)

    assert len(actual_results) == expected_transformers_len
    if expected_transformers_len == 2:
        for transformer, result in actual_results:
            assert str(transformer) in expected_transformers


def test_transformer_normal_transformations_histogram(
        transformer_classification_fitted, data_classification_balanced, numerical_features, seed
):
    """Testing if histogram data is calculated for expected features and normal transformers."""
    expected_keys = {"Price", "Height"}
    X, y = data_classification_balanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=seed)
    train = pd.concat([X_train, y_train], axis=1)[numerical_features]
    test = pd.concat([X_test, y_test], axis=1)[numerical_features]

    actual_results = transformer_classification_fitted.normal_transformations_histograms(train, test)

    assert set(actual_results.keys()) == expected_keys
    for key, item in actual_results.items():
        assert len(item) == 3
