import copy

import seaborn as sns
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder


class DataExplainer:

    numerical = "numerical"
    categorical = "categorical"
    date = "date"

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.full_table = pd.concat([self.X, self.y], axis=1)
        self.columns = self._analyze_columns()
        self.numerical_columns = self.columns["columns"][self.numerical]
        self.categorical_columns = self.columns["columns"][self.categorical]
        self.date_columns = self.columns["columns"][self.date]

    def analyze(self):
        output = {
            "columns": self.columns,
            "figures": self._create_figures(),
            "tables": self._create_tables(),
            "lists": self._create_lists()
        }
        return output

    def _analyze_columns(self):
        columns = [
            self.numerical,
            self.categorical,
            self.date
        ]
        output = {key: list() for key in columns}

        for col in self.X.columns:
            result = self.__column_type(col)
            output[result].append(col)

        output = {key: sorted(output[key]) for key in output}

        x_cols = copy.copy(output)

        y_col_name = self.y.name
        output[self.__column_type(y_col_name)].append(y_col_name)
        xy_cols = output

        return {
            "columns": xy_cols,
            "columns_without_target": x_cols
        }

    def __column_type(self, col):
        if self.full_table[col].dtype == "bool":
            return self.categorical
        else:
            try:
                self.full_table[col].astype("float64")
                return self.numerical
            except TypeError:
                return self.date
            except ValueError:
                return self.categorical
            except:
                raise

    def _create_tables(self):
        d = {
            "described_numeric": self._numeric_describe(),
            "described_categorical": self._categorical_describe(),
            "head": self.__create_df_head()
        }
        return d

    def _numeric_describe(self):
        ds = self.full_table[self.numerical_columns].describe().astype("float64").T
        ds["missing"] = self.full_table.isna().sum() / self.full_table.count()
        return ds

    def _categorical_describe(self):
        # TODO: decide if encoder is needed
        # TODO: decide what to do with NaN values
        df = self.full_table[self.categorical_columns].fillna("NAN")
        enc = OrdinalEncoder()
        new_df = pd.DataFrame(data=enc.fit_transform(df), columns=df.columns)
        return new_df.describe().T

    def __create_df_head(self):
        cols = sorted(self.full_table.columns)
        return self.full_table[cols].head().T

    def _create_lists(self):
        return {
            "unused_cols": self._unused_cols()
        }

    def _unused_cols(self):
        return self.date_columns

    def _create_figures(self):
        d = {
            "pairplot": self.__create_pairplot()
        }
        return d

    def __create_pairplot(self):
        cols = sorted(self.full_table.columns)
        p = sns.pairplot(self.full_table[cols])
        return p
