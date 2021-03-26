from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

from .features import CategoricalFeature

class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    # https://scikit-learn.org/stable/modules/compose.html

    def __init__(self, features):

        self.transformed_X = None
        self.transformed_y = None

        self.features = features
        self.feature_names = features.features(drop_target=True)
        self.target = features.target

        self.preprocessor_X = self._create_preprocessor_X()
        self.preprocessor_y = self._create_preprocessor_y()

    # methods exposed to be compatible with general API
    def fit(self, X):
        self.preprocessor_X = self.preprocessor_X.fit(X)
        return self

    def transform(self, X):
        transformed = self.preprocessor_X.transform(X)
        self.transformed_X = transformed
        return transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def fit_y(self, y):
        self.preprocessor_y = self.preprocessor_y.fit(y)
        return self

    def transform_y(self, y):
        transformed = self.preprocessor_y.transform(y)
        self.transformed_y = transformed
        return transformed

    def fit_transform_y(self, y):
        self.fit_y(y)
        return self.transform_y(y)

    def _create_preprocessor_X(self):

        numerical_features = self.features.numerical_features(drop_target=True)
        categorical_features = self.features.categorical_features(drop_target=True)

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
            QuantileTransformer(output_distribution="normal"),
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
        if isinstance(self.features[self.target], CategoricalFeature):
            transformer = LabelEncoder()
        else:
            # in regression model finder wraps its object in TargetRegressor
            transformer = FunctionTransformer(lambda x: x)

        return transformer
