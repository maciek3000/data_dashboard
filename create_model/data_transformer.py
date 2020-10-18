from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

class Transformer:
    """Wrapper for pipeline for transformations of input and output data."""

    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.pipeline = None
        self.transformed_X = None
        self.transformed_y = None

        self.numerical_columns = None
        self.categorical_columns = None
        self.date_columns = None

        self.__analyze_data()

    def transform(self):

        if not self.pipeline:
            self.__create_pipeline()

    def __analyze_data(self):

        num_cols = []
        date_cols = []
        cat_cols = []

        for col in self.X.columns:

            if self.X[col].dtype == "bool":
                cat_cols.append(col)
            else:
                try:
                    self.X[col].astype("float64")
                    num_cols.append(col)
                except TypeError:
                    date_cols.append(col)
                except ValueError:
                    cat_cols.append(col)
                except:
                    raise

        self.numerical_columns = num_cols
        self.date_columns = date_cols
        self.categorical_columns = cat_cols

    def __create_pipeline(self):
        imputer = self.__create_imputer()


    def __create_imputer(self):
        return SimpleImputer(strategy="most_frequent")

