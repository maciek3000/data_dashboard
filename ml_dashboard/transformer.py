from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, FunctionTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np
import pandas as pd
import warnings

from .model_finder import WrappedModelRegression
from .functions import calculate_numerical_bins, modify_histogram_edges


class WrapperFunctionTransformer:

    def __init__(self, str_repr, func_transformer):
        self.transformer = func_transformer
        self.str_repr = str_repr

    def fit(self, *args, **kwargs):
        self.transformer.fit(*args, **kwargs)
        return self

    def fit_transform(self, *args, **kwargs):
        return self.transformer.fit_transform(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        return self.transformer.get_params(*args, **kwargs)

    def inverse_transform(self, *args, **kwargs):
        return self.transformer.inverse_transform(*args, **kwargs)

    def set_params(self, *args, **kwargs):
        self.transformer.set_params(*args, **kwargs)
        return self

    def transform(self, *args, **kwargs):
        return self.transformer.transform(*args, **kwargs)

    def __str__(self):
        return str(self.str_repr)


class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    _default_categorical_transformers = [
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ]
    _default_numerical_transformers = [
        SimpleImputer(strategy="median"),
        QuantileTransformer(output_distribution="normal"),
        StandardScaler()
    ]

    _default_transformers_random_state = [QuantileTransformer]

    _categorical_pipeline = "categorical"
    _numerical_pipeline = "numerical"
    _one_hot_encoder_tr_name = "onehotencoder"
    _normal_transformers = [QuantileTransformer, PowerTransformer]

    _func_trans_classification_pos_label_text = "FunctionTransformer(classification_pos_label='{}')"
    _func_trans_regression_wrapper_text = "TransformedTargetRegressor(transformer=QuantileTransformer(output_distribution='normal'))"

    def __init__(self, categorical_features, numerical_features, target_type, random_state=None, classification_pos_label=None):

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_type = target_type
        self.random_state = random_state
        self.classification_pos_label = classification_pos_label

        self.categorical_transformers = self._check_random_state(self._default_categorical_transformers)
        self.numerical_transformers = self._check_random_state(self._default_numerical_transformers)

        # self.features = features
        # self.feature_names = features.features(drop_target=True)
        # self.target = features.target

        self.preprocessor_X = self._create_preprocessor_X()
        self.preprocessor_y = self._create_preprocessor_y()

    # methods exposed to be compatible with general API
    def fit(self, X):
        self.preprocessor_X = self.preprocessor_X.fit(X)
        return self

    def transform(self, X):
        transformed = self.preprocessor_X.transform(X)
        return transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit_y(self, y):
        self.preprocessor_y = self.preprocessor_y.fit(y)
        return self

    def transform_y(self, y):
        transformed = self.preprocessor_y.transform(y)
        return transformed

    def fit_transform_y(self, y):
        self.fit_y(y)
        return self.transform_y(y)

    def transformations(self):
        output = {}
        new_cols = self.transformed_columns()
        for feature in (self.categorical_features + self.numerical_features):
            transformers = self.transformers(feature)
            transformed_cols = [col for col in new_cols if col.startswith(feature)]
            output[feature] = (transformers, transformed_cols)

        return output

    def y_transformers(self):
        transformer = self.preprocessor_y
        return [transformer]

    def transformers(self, feature):
        if feature in self.categorical_features:
            return self.categorical_transformers
        elif feature in self.numerical_features:
            return self.numerical_transformers
        else:
            return None

    def transformed_columns(self):
        # TODO: check if transformer is fitted
        # checking if there are any categorical_features at all
        if len(self.categorical_features) > 0:
            cat_transformers = self.preprocessor_X.named_transformers_[self._categorical_pipeline]
            try:
                categorical_cols = cat_transformers[self._one_hot_encoder_tr_name].get_feature_names(self.categorical_features).tolist()
            except KeyError:
                categorical_cols = self.categorical_features
        else:
            categorical_cols = []
        output = self.numerical_features + categorical_cols
        return output

    def normal_transformations_histograms(self, feature_train_data, feature_test_data):
        output = {}
        for feature in feature_train_data.columns:
            train_series = feature_train_data[feature].to_numpy().reshape(-1, 1)
            test_series = feature_test_data[feature].to_numpy().reshape(-1, 1)
            transformations = self.normal_transformations(train_series, test_series)

            histogram_data = []
            for transformer, transformed_series in transformations:
                series = pd.Series(transformed_series.reshape(1, -1)[0])
                bins = calculate_numerical_bins(series)
                hist, edges = np.histogram(series, density=True, bins=bins)
                left_edges, right_edges = modify_histogram_edges(edges)
                result = (transformer, (hist, left_edges, right_edges))
                histogram_data.append(result)

            output[feature] = histogram_data

        return output

    def normal_transformations(self, feature_train_data, feature_test_data):

        normal_transformers = [
            QuantileTransformer(output_distribution="normal", random_state=self.random_state),
            PowerTransformer(method="yeo-johnson")
        ]

        # box-cox works only with positive values, 0s excluded
        nmin = np.nanmin
        if nmin(feature_train_data) > 0 and nmin(feature_test_data) > 0:
            normal_transformers.append(PowerTransformer(method="box-cox"))

        output = []
        for transformer in normal_transformers:
            pipe = make_pipeline(
                SimpleImputer(strategy="median"),
                transformer
            )
            # Catching warnings, mostly from QuantileTransformer
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(feature_train_data)
            output.append((transformer, pipe.transform(feature_test_data)))

        return output

    def set_custom_preprocessor_X(self, numerical_transformers, categorical_transformers):
        self.numerical_transformers = numerical_transformers
        self.categorical_transformers = categorical_transformers
        self.preprocessor_X = self._create_preprocessor_X()

    def set_custom_preprocessor_y(self, transformer):
        self.preprocessor_y = transformer

    def _create_preprocessor_X(self):

        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        # https://scikit-learn.org/stable/modules/compose.html

        numerical_features = self.numerical_features  # self.features.numerical_features(drop_target=True)
        categorical_features = self.categorical_features  # self.features.categorical_features(drop_target=True)

        numeric_transformer = make_pipeline(*self.numerical_transformers)
        categorical_transformer = make_pipeline(*self.categorical_transformers)

        col_transformer = ColumnTransformer(
            transformers=[
                (self._numerical_pipeline, numeric_transformer, numerical_features),
                (self._categorical_pipeline, categorical_transformer, categorical_features)
            ]
        )

        return col_transformer

    def _create_preprocessor_y(self):
        if self.target_type == "Categorical":
            if self.classification_pos_label is not None:
                func_transformer = FunctionTransformer(lambda x: np.where(x == self.classification_pos_label, 1, 0))
                text = self._func_trans_classification_pos_label_text.format(self.classification_pos_label)
                transformer = WrapperFunctionTransformer(text, func_transformer)
            else:
                transformer = LabelEncoder()
        elif self.target_type == "Numerical":
            # in regression model finder wraps its object in TargetRegressor
            func_transformer = FunctionTransformer(func=None)  # identity function, x = x
            text = self._func_trans_regression_wrapper_text
            transformer = WrapperFunctionTransformer(text, func_transformer)
        else:
            raise ValueError(
                "Target type set as {target_type}, should be Categorical or Numerical".format(target_type=self.target_type)
            )

        return transformer

    def y_classes(self):
        if self.target_type == "Numerical":
            raise ValueError("No classes present in regression problem.")

        return self.preprocessor_y.classes_

    def _check_random_state(self, default_transformers):
        for transformer in default_transformers:
            if transformer.__class__ in self._default_transformers_random_state:
                transformer.random_state = self.random_state

        return default_transformers
