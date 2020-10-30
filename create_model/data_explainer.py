import copy

import seaborn as sns
import pandas as pd


class DataExplainer:

    key_cols = "columns"
    key_cols_wo_target = "columns_without_target"
    key_figs = "figures"
    key_tables = "tables"
    key_lists = "lists"

    key_numerical = "numerical"
    key_categorical = "categorical"
    key_date = "date"

    plot_text_color = "#8C8C8C"
    plot_color = "#19529c"

    def __init__(self, X, y):
        sns.set_style("white", {
            "axes.edgecolor": self.plot_text_color,
            "axes.labelcolor": self.plot_text_color,
            "text.color": self.plot_text_color,
            "font.sans-serif": ["Lato"],
            "xtick.color": self.plot_text_color,
            "ytick.color": self.plot_text_color,
        })

        self.X = X
        self.y = y
        self.full_table = pd.concat([self.X, self.y], axis=1)

        self.columns = self._analyze_columns()

        self.numerical_columns = self.columns[self.key_cols][self.key_numerical]
        self.categorical_columns = self.columns[self.key_cols][self.key_categorical]
        self.date_columns = self.columns[self.key_cols][self.key_date]

    def analyze(self):
        output = {
            self.key_cols: self.columns,
            self.key_figs: self._create_figures(),
            self.key_tables: self._create_tables(),
            self.key_lists: self._create_lists()
        }
        return output

    def _analyze_columns(self):
        columns = [
            self.key_numerical,
            self.key_categorical,
            self.key_date
        ]
        output = {key: list() for key in columns}

        for col in self.X.columns:
            result = self.__column_type(col)
            output[result].append(col)

        x_cols = output
        xy_cols = copy.deepcopy(output)  # deepcopy is needed to not overwrite x_cols dict

        y_col_name = self.y.name
        xy_cols[self.__column_type(y_col_name)].append(y_col_name)

        x_cols = {key: sorted(x_cols[key], key=str.upper) for key in x_cols}
        xy_cols = {key: sorted(xy_cols[key], key=str.upper) for key in xy_cols}

        return {
            self.key_cols: xy_cols,
            self.key_cols_wo_target: x_cols
        }

    def __column_type(self, col):
        if self.full_table[col].dtype == "bool":
            return self.key_numerical
        else:
            try:
                self.full_table[col].astype("float64")
                return self.key_numerical
            except TypeError:
                return self.key_date
            except ValueError:
                return self.key_categorical
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
        ds["missing"] = self.full_table.isna().sum() / max(self.full_table.count())
        return ds

    def _categorical_describe(self):
        df = self._categorical_to_ordinal(self.full_table[self.categorical_columns]).describe().astype("float64").T
        df["missing"] = self.full_table.isna().sum() / max(self.full_table.count())
        return df

    def _categorical_to_ordinal(self, df):
        _ = {}
        for col in df.columns:
            vals = df[col].unique()
            mapped = {val: x+1 for val, x in zip(vals, range(len(vals))) if not pd.isna(val)}  # count starts at 1
            _[col] = mapped

        new_df = df.replace(_)
        return new_df

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
        num = self.full_table[self.numerical_columns]
        cat = self.full_table[self.categorical_columns]
        df = pd.concat([num, self._categorical_to_ordinal(cat)], axis=1)
        df = df[sorted(df.columns)]
        colors = {"color": self.plot_color}
        p = sns.pairplot(df, plot_kws=colors, diag_kws=colors)
        return p
