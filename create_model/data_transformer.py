from sklearn.pipeline import make_pipeline

import pandas as pd

class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.pipeline = None
        self.transformed_X = None
        self.transformed_y = None

        self.numerical_columns = None
        self.categorical_columns = None

        self.__analyze_data()

    def transform(self):

        if not self.pipeline:
            self.__create_pipeline()

    def __analyze_data(self):

        num_cols = []
        date_cols = []
        cat_cols = []

        for col in self.X.columns:

            # datetime columns
            try:
                self.X[col].astype("datetime64[ns]")
                date_cols.append(col)
            except ValueError:

                # numerical columns
                try:
                    self.X[col].astype("float64")
                    num_cols.append(col)
                except ValueError:
                    pass

                # datetime columns
                # TODO: just a prototype
                try:
                    pd.to_datetime(self.X[col])
                    date_cols.append(col)
                except ValueError as e:
                    pass

        self.numerical_columns = num_cols

    def __create_pipeline(self):
        pass
