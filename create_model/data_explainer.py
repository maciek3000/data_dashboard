import seaborn as sns


class DataExplainer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def analyze(self):
        output = {
            "figures": self._create_figures(),
            "tables": self._create_tables()
        }
        return output

    def _create_tables(self):
        d = {
            "described_numeric": self.__describe_df(),
            "head": self.__create_df_head()
        }
        return d

    def __describe_df(self):
        return self.X.describe()

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
