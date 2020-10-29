import seaborn as sns


class DataExplainer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def analyze(self):
        output = {
            "columns": self._analyze_columns(),
            "figures": self._create_figures(),
            "tables": self._create_tables()
        }
        return output

    def _analyze_columns(self):

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

        cols = {
            "numerical": num_cols,
            "date": date_cols,
            "categorical": cat_cols
        }

        return cols

    def _create_tables(self):
        d = {
            "described_numeric": self.__describe_df(),
            "head": self.__create_df_head()
        }
        return d



    def __describe_df(self):
        return self.X.describe().T

    def __create_df_head(self):
        return self.X.head().T

    def _create_figures(self):
        d = {
            "pairplot": self.__create_pairplot()
        }
        return d

    def __create_pairplot(self):
        p = sns.pairplot(self.X)
        return p
