from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import numpy as np


class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""
    _default_categorical_transformers = [
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ]
    _default_numerical_transformers = [
        SimpleImputer(strategy="median"),
        StandardScaler(),
        QuantileTransformer(output_distribution="normal")
    ]

    _default_transformers_random_state = [QuantileTransformer]

    _categorical_pipeline = "categorical"
    _numerical_pipeline = "numerical"
    _one_hot_encoder_tr_name = "onehotencoder"

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

    def transformers(self, feature):
        if feature in self.categorical_features:
            return self.categorical_transformers
        elif feature in self.numerical_features:
            return self.numerical_transformers
        else:
            return None

    def transformed_columns(self):
        # TODO: check if transformer is fitted
        cat_transformers = self.preprocessor_X.named_transformers_[self._categorical_pipeline]
        try:
            categorical_cols = cat_transformers[self._one_hot_encoder_tr_name].get_feature_names(self.categorical_features).tolist()
        except KeyError:
            categorical_cols = self.categorical_features
        output = self.numerical_features + categorical_cols
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
                transformer = FunctionTransformer(lambda x: np.where(x == self.classification_pos_label, 1, 0))
            else:
                transformer = LabelEncoder()
        elif self.target_type == "Numerical":
            # in regression model finder wraps its object in TargetRegressor
            transformer = FunctionTransformer(lambda x: x)
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
