from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
    # https://scikit-learn.org/stable/modules/compose.html

    def __init__(self, features):

        self.features = features
        self.feature_names = features.features(drop_target=True)
        self.target = features.target

        self.preprocessor = self._create_preprocessor()

    # methods exposed to be compatible with general API
    def fit(self, X):
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _create_preprocessor(self):

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
