from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np


class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    # https://scikit-learn.org/stable/modules/compose.html

    def __init__(self, categorical_features, numerical_features, target_type, random_state=None, classification_pos_label=None):

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_type = target_type
        self.random_state = random_state
        self.classification_pos_label = classification_pos_label

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

    def _create_preprocessor_X(self):

        numerical_features = self.numerical_features  # self.features.numerical_features(drop_target=True)
        categorical_features = self.categorical_features  # self.features.categorical_features(drop_target=True)

        numeric_transformer = self._create_numeric_transformer()
        categorical_transformer = self._create_categorical_transformer()

        col_transformer = ColumnTransformer(
            transformers=[
                ("numerical", numeric_transformer, numerical_features),
                ("categorical", categorical_transformer, categorical_features)
            ]
        )
        return col_transformer

    def _create_numeric_transformer(self):
        transformer = make_pipeline(
            SimpleImputer(strategy="median"),
            QuantileTransformer(output_distribution="normal", random_state=self.random_state),
            StandardScaler()
        )
        return transformer

    def _create_categorical_transformer(self):
        transformer = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore")
        )
        return transformer

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
