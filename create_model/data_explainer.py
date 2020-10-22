import seaborn as sns


class DataExplainer:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def analyze(self):
        pairplot = sns.pairplot(self.X)
        describe = self.X.describe().to_html()
        head = self.X.head().T.to_html()
        output = {
            "figures": {
                "pairplot": pairplot,
            },

            "html": {
                "describe": describe,
                "head": head
            }
        }
        return output
