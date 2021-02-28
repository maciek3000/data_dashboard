from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, QuantileTransformer
from sklearn.impute import SimpleImputer


class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

    def __init__(self, X, y, numerical_columns, categorical_columns):
        self.X = X
        self.y = y

        self.preprocessor = None
        self.transformed_X = None
        self.transformed_y = None

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        # self.numerical_columns = columns["numerical"]
        # self.categorical_columns = columns["categorical"]
        # self.date_columns = columns["date"]

        self.preprocessor = self._create_preprocessor()

        # placeholder
        self.data_objects = None

    def fit(self):
        self.preprocessor = self.preprocessor.fit(self.X)
        self.update_data_objects()
        return self

    def transform(self, X=None):
        if X is None:
            X = self.X

        # TODO: transformer should transform y as well when applicable
        # however, API shouldn't be broken here otherwise pipeline will stop working
        return self.preprocessor.transform(X)

    def _create_preprocessor(self):
        numeric_transformer = self._create_numeric_transformer()
        categorical_transformer = self._create_categorical_transformer()

        col_transformer = ColumnTransformer(
            transformers=[
                ("numerical", numeric_transformer, self.numerical_columns),
                ("categorical", categorical_transformer, self.categorical_columns)
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

    def update_data_objects(self):
        # placeholder
        self.data_objects = {"transformations": None}